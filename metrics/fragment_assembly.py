"""Evaluation metrics for Task 5: Fragment-to-Lead Assembly.

Functions:
- assess_fragment_incorporation: check that the agent's molecule contains all required fragments
- assess_fragment_linking: check that all fragments are covalently connected
- assess_boltz_task5: run Boltz co-fold, check binding probability, affinity, and fragment 3D pose RMSD
"""

import os
import json
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from metrics.boltz_manager import BoltzManager
from metrics.lead_optimization import read_smiles_file


def assess_fragment_incorporation(agent_file_path, fragment_smarts):
    """Check whether the agent's molecule contains all required fragment substructures.

    Args:
        agent_file_path: path to file containing agent's SMILES
        fragment_smarts: list of SMARTS strings, one per fragment

    Returns:
        dict with "all_passed" bool and "details" list
    """
    smiles = read_smiles_file(agent_file_path)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "all_passed": False,
            "details": [{"fragment_idx": i, "smarts": s, "matched": False,
                         "reason": "Cannot parse agent SMILES"}
                        for i, s in enumerate(fragment_smarts)],
        }

    details = []
    for i, smarts in enumerate(fragment_smarts):
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            details.append({
                "fragment_idx": i, "smarts": smarts, "matched": False,
                "reason": "Invalid SMARTS pattern",
            })
            continue
        matched = mol.HasSubstructMatch(pattern)
        details.append({
            "fragment_idx": i, "smarts": smarts, "matched": matched,
        })

    all_passed = all(d["matched"] for d in details)
    return {"all_passed": all_passed, "details": details}


def _find_disjoint_matches(all_matches, idx=0, used_atoms=set()):
    """Find a combination of substructure matches where all fragments use disjoint atoms.

    Args:
        all_matches: list of lists, where all_matches[i] is a list of atom-index tuples
                     for fragment i (from GetSubstructMatches)
        idx: current fragment index being assigned
        used_atoms: set of atom indices already claimed by previously assigned fragments

    Returns:
        list of sets (one per fragment) if a disjoint assignment exists, else None
    """
    if idx == len(all_matches):
        return []
    for match in all_matches[idx]:
        match_set = set(match)
        if match_set & used_atoms:
            continue
        rest = _find_disjoint_matches(all_matches, idx + 1, used_atoms | match_set)
        if rest is not None:
            return [match_set] + rest
    return None


def assess_fragment_linking(agent_file_path, fragment_smarts):
    """Check that all fragments are present as distinct, covalently connected substructures.

    Finds a disjoint assignment of substructure matches across all fragment SMARTS
    patterns, then verifies a covalent path exists between each pair.

    Args:
        agent_file_path: path to file containing agent's SMILES
        fragment_smarts: list of SMARTS strings (must be >= 2)

    Returns:
        dict with "all_passed" bool and "reason" on failure
    """
    smiles = read_smiles_file(agent_file_path)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"all_passed": False, "reason": "Cannot parse agent SMILES"}

    if len(fragment_smarts) < 2:
        return {"all_passed": True, "reason": "Single fragment, linking check not applicable"}

    # Collect all possible matches for each fragment
    all_matches = []
    for i, smarts in enumerate(fragment_smarts):
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            return {"all_passed": False, "reason": f"Invalid SMARTS for fragment {i}"}
        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            return {"all_passed": False, "reason": f"Fragment {i} not found in molecule"}
        all_matches.append(matches)

    # Find a disjoint assignment of matches
    fragment_atoms = _find_disjoint_matches(all_matches)
    if fragment_atoms is None:
        return {
            "all_passed": False,
            "reason": "Fragments cannot be matched to distinct (non-overlapping) substructures",
        }

    # Verify covalent path exists between each pair of fragments
    for i in range(len(fragment_atoms)):
        for j in range(i + 1, len(fragment_atoms)):
            atom_i = next(iter(fragment_atoms[i]))
            atom_j = next(iter(fragment_atoms[j]))
            path = Chem.GetShortestPath(mol, atom_i, atom_j)
            if not path:
                return {
                    "all_passed": False,
                    "reason": f"No path between fragment {i} and fragment {j}",
                }

    return {"all_passed": True}


def _compute_fragment_rmsd(docked_mol, fragment_sdf_path, smarts):
    """Compute RMSD between fragment atoms in docked molecule and input fragment SDF.

    Args:
        docked_mol: RDKit Mol with 3D coords (the docked ligand, already aligned)
        fragment_sdf_path: path to the input fragment SDF file
        smarts: SMARTS pattern for the fragment

    Returns:
        float RMSD or None if matching fails
    """
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return None

    # Get fragment reference coords from SDF
    supplier = Chem.SDMolSupplier(fragment_sdf_path, removeHs=True)
    frag_mol = next(supplier, None)
    if frag_mol is None:
        return None

    # Match SMARTS on the fragment SDF to get atom ordering
    frag_matches = frag_mol.GetSubstructMatches(pattern)
    if not frag_matches:
        return None
    frag_match = frag_matches[0]
    frag_conf = frag_mol.GetConformer()
    frag_coords = np.array([frag_conf.GetAtomPosition(idx) for idx in frag_match])

    # Match SMARTS on docked ligand — take the match with lowest RMSD
    # Strip hydrogens for matching
    docked_noH = Chem.RemoveHs(docked_mol)
    docked_matches = docked_noH.GetSubstructMatches(pattern)
    if not docked_matches:
        return None

    docked_conf = docked_noH.GetConformer()
    best_rmsd = float("inf")
    for match in docked_matches:
        docked_coords = np.array([docked_conf.GetAtomPosition(idx) for idx in match])
        if len(docked_coords) != len(frag_coords):
            continue
        diff = docked_coords - frag_coords
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        best_rmsd = min(best_rmsd, rmsd)

    return best_rmsd if best_rmsd < float("inf") else None


def _kabsch_align(mobile_coords, target_coords):
    """Compute optimal rotation+translation to align mobile onto target (Kabsch algorithm).

    Args:
        mobile_coords: Nx3 numpy array
        target_coords: Nx3 numpy array

    Returns:
        (rotation 3x3, translation 3x1) to apply as: aligned = (mobile - mobile_centroid) @ R + target_centroid
    """
    mobile_centroid = mobile_coords.mean(axis=0)
    target_centroid = target_coords.mean(axis=0)

    mobile_centered = mobile_coords - mobile_centroid
    target_centered = target_coords - target_centroid

    H = mobile_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    return R, mobile_centroid, target_centroid


def _align_boltz_to_pocket(boltz_cif_path, pocket_pdb_path, pocket_contacts=None):
    """Align Boltz output protein CAs to pocket.pdb CAs via Kabsch.

    Args:
        boltz_cif_path: path to Boltz CIF file
        pocket_pdb_path: path to pocket PDB file
        pocket_contacts: list of [chain, res_id] pairs. If provided, only these
            residues are used for alignment. Otherwise all matching CAs are used.

    Returns (rotation, mobile_centroid, target_centroid) or None if alignment fails.
    """
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io.pdb as pdb_io

    # Load Boltz structure
    boltz_struct = pdbx.get_structure(pdbx.CIFFile.read(boltz_cif_path), model=1, use_author_fields=True)
    boltz_protein = boltz_struct[boltz_struct.chain_id != "LIG"]
    boltz_ca = boltz_protein[(boltz_protein.atom_name == "CA") & (boltz_protein.element == "C")]

    # Load pocket PDB
    pocket_struct = pdb_io.PDBFile.read(pocket_pdb_path).get_structure(model=1)
    pocket_ca = pocket_struct[(pocket_struct.atom_name == "CA") & (pocket_struct.element == "C")]

    # Match CAs by (chain_id, res_id)
    boltz_keys = {}
    for i in range(len(boltz_ca)):
        key = (boltz_ca.chain_id[i], boltz_ca.res_id[i])
        boltz_keys[key] = boltz_ca.coord[i]

    pocket_keys = {}
    for i in range(len(pocket_ca)):
        key = (pocket_ca.chain_id[i], pocket_ca.res_id[i])
        pocket_keys[key] = pocket_ca.coord[i]

    if pocket_contacts is not None:
        # Only use pocket contact residues for alignment
        contact_set = set((chain, res_id) for chain, res_id in pocket_contacts)
        common_keys = sorted(contact_set & set(boltz_keys.keys()) & set(pocket_keys.keys()))
    else:
        common_keys = sorted(set(boltz_keys.keys()) & set(pocket_keys.keys()))

    if len(common_keys) < 3:
        return None

    mobile = np.array([boltz_keys[k] for k in common_keys])
    target = np.array([pocket_keys[k] for k in common_keys])

    return _kabsch_align(mobile, target)


def _select_best_aligned_model(predictions_dir, pocket_pdb_path, job_name, pocket_contacts=None):
    """Select the Boltz model whose pocket CAs best align to pocket.pdb.

    Returns the model index with the lowest pocket CA RMSD after Kabsch superposition.
    """
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure.io.pdbx as pdbx

    # Load pocket CAs once
    pocket_struct = pdb_io.PDBFile.read(pocket_pdb_path).get_structure(model=1)
    pocket_ca = pocket_struct[(pocket_struct.atom_name == "CA") & (pocket_struct.element == "C")]
    pocket_keys = {}
    for i in range(len(pocket_ca)):
        pocket_keys[(pocket_ca.chain_id[i], pocket_ca.res_id[i])] = pocket_ca.coord[i]

    contact_set = None
    if pocket_contacts is not None:
        contact_set = set((chain, res_id) for chain, res_id in pocket_contacts)

    best_idx = 0
    best_rmsd = float("inf")

    cif_files = sorted(f for f in os.listdir(predictions_dir)
                       if f.startswith(f"{job_name}_model_") and f.endswith(".cif"))

    for cif_file in cif_files:
        model_idx = int(cif_file.split("_model_")[1].split(".")[0])
        cif_path = os.path.join(predictions_dir, cif_file)

        boltz_struct = pdbx.get_structure(
            pdbx.CIFFile.read(cif_path), model=1, use_author_fields=True
        )
        boltz_protein = boltz_struct[boltz_struct.chain_id != "LIG"]
        boltz_ca = boltz_protein[(boltz_protein.atom_name == "CA") & (boltz_protein.element == "C")]

        boltz_keys = {}
        for i in range(len(boltz_ca)):
            boltz_keys[(boltz_ca.chain_id[i], boltz_ca.res_id[i])] = boltz_ca.coord[i]

        if contact_set is not None:
            common_keys = sorted(contact_set & set(boltz_keys.keys()) & set(pocket_keys.keys()))
        else:
            common_keys = sorted(set(boltz_keys.keys()) & set(pocket_keys.keys()))
        if len(common_keys) < 3:
            continue

        mobile = np.array([boltz_keys[k] for k in common_keys])
        target = np.array([pocket_keys[k] for k in common_keys])

        R, mc, tc = _kabsch_align(mobile, target)
        aligned = (mobile - mc) @ R + tc
        ca_rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))

        if ca_rmsd < best_rmsd:
            best_rmsd = ca_rmsd
            best_idx = model_idx

    return best_idx


def assess_boltz_task5(agent_file_path, protein_sequences_path, pocket_contacts,
                       diffusion_samples=10, fragment_sdf_paths=None,
                       fragment_smarts=None, witness_affinity_pred_value=None,
                       pocket_pdb_path=None):
    """Run Boltz co-fold for task 5 and check binding, affinity, and fragment RMSD.

    Args:
        agent_file_path: path to file containing agent's SMILES
        protein_sequences_path: path to protein_sequences.json (key: "target")
        pocket_contacts: list of [chain, seqres_index] pairs
        diffusion_samples: number of diffusion samples
        fragment_sdf_paths: list of paths to input fragment SDF files (for RMSD check)
        fragment_smarts: list of SMARTS strings (for RMSD check)
        witness_affinity_pred_value: witness log10(IC50) value (for affinity threshold)
        pocket_pdb_path: path to pocket.pdb (for alignment before RMSD check)

    Returns:
        dict with binding_probability, affinity, RMSD results, and all_passed
    """
    smiles = read_smiles_file(agent_file_path)

    with open(protein_sequences_path, 'r') as f:
        all_sequences = json.load(f)
    protein_sequences = all_sequences["target"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = BoltzManager(work_dir=tmp_dir)

        yaml_path = manager.create_input_yaml(
            protein_sequences,
            smiles,
            job_name="agent_dock",
            pocket_contacts=pocket_contacts,
            predict_affinity=True,
        )
        results_dir = manager.run_prediction(yaml_path, diffusion_samples=diffusion_samples)
        agent_boltz_dir = os.path.join(results_dir, "boltz_results_agent_dock")

        # Binding probability check
        binding_prob = manager.get_mean_affinity(
            agent_boltz_dir, "agent_dock", key="affinity_probability_binary"
        )
        binding_prob_passed = binding_prob > 0.7

        # Affinity check (witness-relative)
        affinity_pred = manager.get_mean_affinity(
            agent_boltz_dir, "agent_dock", key="affinity_pred_value"
        )

        affinity_passed = True
        affinity_threshold = None
        if witness_affinity_pred_value is not None:
            affinity_threshold = witness_affinity_pred_value + 0.1
            affinity_passed = affinity_pred < affinity_threshold

        # Fragment 3D RMSD check
        rmsd_per_fragment = []
        rmsd_passed = True

        if fragment_sdf_paths and fragment_smarts and pocket_pdb_path:
            # Select the model with lowest protein CA RMSD to pocket.pdb
            predictions_dir = os.path.join(agent_boltz_dir, "predictions", "agent_dock")
            best_model_idx = _select_best_aligned_model(predictions_dir, pocket_pdb_path, "agent_dock", pocket_contacts=pocket_contacts)
            best_cif = os.path.join(predictions_dir, f"agent_dock_model_{best_model_idx}.cif")

            # Align Boltz output to pocket reference frame using pocket contact residues
            alignment = _align_boltz_to_pocket(best_cif, pocket_pdb_path, pocket_contacts=pocket_contacts)

            # Extract docked ligand
            docked_mol = manager.extract_ligand_from_cif(best_cif, template_smiles=smiles)

            if docked_mol is not None and alignment is not None:
                R, mobile_centroid, target_centroid = alignment

                # Apply alignment transform to ligand atom coordinates
                docked_noH = Chem.RemoveHs(docked_mol)
                conf = docked_noH.GetConformer()
                for atom_idx in range(docked_noH.GetNumAtoms()):
                    pos = np.array(conf.GetAtomPosition(atom_idx))
                    aligned_pos = (pos - mobile_centroid) @ R + target_centroid
                    conf.SetAtomPosition(atom_idx, aligned_pos.tolist())

                # Also align the H-containing version for RMSD computation
                conf_H = docked_mol.GetConformer()
                for atom_idx in range(docked_mol.GetNumAtoms()):
                    pos = np.array(conf_H.GetAtomPosition(atom_idx))
                    aligned_pos = (pos - mobile_centroid) @ R + target_centroid
                    conf_H.SetAtomPosition(atom_idx, aligned_pos.tolist())

                for frag_sdf, frag_smarts in zip(fragment_sdf_paths, fragment_smarts):
                    rmsd = _compute_fragment_rmsd(docked_mol, frag_sdf, frag_smarts)
                    rmsd_per_fragment.append(round(rmsd, 4) if rmsd is not None else None)

                rmsd_passed = all(
                    r is not None and r < 2.0 for r in rmsd_per_fragment
                )
            else:
                rmsd_per_fragment = [None] * len(fragment_sdf_paths)
                rmsd_passed = False

    all_passed = binding_prob_passed and affinity_passed and rmsd_passed

    result = {
        "binding_probability": round(binding_prob, 4),
        "binding_prob_passed": binding_prob_passed,
        "affinity_pred_value": round(affinity_pred, 4),
        "affinity_passed": affinity_passed,
        "rmsd_per_fragment": rmsd_per_fragment,
        "rmsd_passed": rmsd_passed,
        "all_passed": all_passed,
    }
    if affinity_threshold is not None:
        result["affinity_threshold"] = round(affinity_threshold, 4)

    return result
