import os
import tempfile
import numpy as np
import json
from collections import Counter
from rdkit import Chem
from metrics.boltz_manager import BoltzManager
from metrics.shape import _load_mol_from_file

from plip.structure.preparation import PDBComplex

# Map specific PLIP interaction types to broad categories.
# This coarsening makes consensus fingerprints more robust to stochastic
# variation in diffusion samples (e.g. donor/acceptor swap on the same H-bond).
BROAD_TYPE_MAP = {
    "HBond_ProtDonor": "HBond",
    "HBond_LigDonor": "HBond",
    "PiStacking": "Aromatic",
    "PiCation_LigAro": "Aromatic",
    "PiCation_ProtAro": "Aromatic",
    "SaltBridge_LigNeg": "Ionic",
    "SaltBridge_ProtNeg": "Ionic",
    "HalogenBond": "Halogen",
    "Metal": "Metal",
    "Hydrophobic": "Hydrophobic",
}

def run_plip_analysis(lig_pdb_block, prot_pdb_block):
    """
    Runs PLIP analysis on a complex formed by ligand and protein PDB blocks.
    Returns a set of coarsened interaction tuples: (chain, resnr, broad_type).
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_pdb:
        if prot_pdb_block:
            tmp_pdb.write(prot_pdb_block)
            if not prot_pdb_block.endswith('\n'):
                tmp_pdb.write('\n')

        if lig_pdb_block:
            tmp_pdb.write(lig_pdb_block)
            if not lig_pdb_block.endswith('\n'):
                tmp_pdb.write('\n')

        complex_path = tmp_pdb.name

    interactions_set = set()

    try:
        mol_complex = PDBComplex()
        mol_complex.load_pdb(complex_path)
        mol_complex.analyze()

        for lig_key, interaction in mol_complex.interaction_sets.items():
            def add_interactions(int_list, int_type):
                broad_type = BROAD_TYPE_MAP[int_type]
                for i in int_list:
                    key = (i.reschain, i.resnr, broad_type)
                    interactions_set.add(key)

            add_interactions(interaction.hydrophobic_contacts, "Hydrophobic")
            add_interactions(interaction.hbonds_pdon, "HBond_ProtDonor")
            add_interactions(interaction.hbonds_ldon, "HBond_LigDonor")
            add_interactions(interaction.pistacking, "PiStacking")
            add_interactions(interaction.pication_laro, "PiCation_LigAro")
            add_interactions(interaction.pication_paro, "PiCation_ProtAro")
            add_interactions(interaction.saltbridge_lneg, "SaltBridge_LigNeg")
            add_interactions(interaction.saltbridge_pneg, "SaltBridge_ProtNeg")
            add_interactions(interaction.halogen_bonds, "HalogenBond")
            add_interactions(interaction.metal_complexes, "Metal")

    except Exception as e:
        print(f"PLIP Analysis Error: {e}")
    finally:
        if os.path.exists(complex_path):
            os.remove(complex_path)

    return interactions_set

def build_consensus_fingerprint(pdb_block_pairs, threshold=0.3):
    """Build a consensus interaction fingerprint from multiple diffusion samples.

    Args:
        pdb_block_pairs: list of (lig_pdb_block, prot_pdb_block) tuples
        threshold: fraction of samples an interaction must appear in to be kept

    Returns:
        set of (chain, resnr, broad_type) tuples that meet the threshold
    """
    n_samples = len(pdb_block_pairs)
    if n_samples == 0:
        return set()

    counts = Counter()
    for lig_pdb, prot_pdb in pdb_block_pairs:
        sample_interactions = run_plip_analysis(lig_pdb, prot_pdb)
        for interaction in sample_interactions:
            counts[interaction] += 1

    consensus = set()
    for interaction, count in counts.items():
        if count / n_samples >= threshold:
            consensus.add(interaction)

    return consensus

def calculate_consensus_tanimoto(agent_pdb_pairs, ref_pdb_pairs, threshold=0.3):
    """Calculate Tanimoto similarity between consensus interaction fingerprints.

    Args:
        agent_pdb_pairs: list of (lig_pdb_block, prot_pdb_block) for agent samples
        ref_pdb_pairs: list of (lig_pdb_block, prot_pdb_block) for reference samples
        threshold: consensus threshold fraction

    Returns:
        Tanimoto similarity (float)
    """
    agent_fp = build_consensus_fingerprint(agent_pdb_pairs, threshold)
    ref_fp = build_consensus_fingerprint(ref_pdb_pairs, threshold)

    intersection = len(agent_fp & ref_fp)
    union = len(agent_fp | ref_fp)

    return intersection / union if union > 0 else 0.0

def assess_boltz_all(agent_file_path, protein_fasta_path, ground_truth_ligand_path, ground_truth_protein_path, pocket_contacts=None, co_ligands=None, diffusion_samples=10):
    """
    Consolidated metric that runs Boltz once and returns both interaction similarity and binding probability.

    Uses consensus interaction fingerprints (threshold=30%) across all diffusion samples
    for both agent and reference, and reports median binding probability.

    Args:
        co_ligands: optional list of {"id": ..., "smiles": ...} dicts for additional ligands
                    present in the co-crystallized structure. These are included in the Boltz
                    co-folding alongside the agent's molecule (which replaces the target ligand).
        diffusion_samples: number of Boltz diffusion samples to generate for the agent molecule.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = BoltzManager(work_dir=tmp_dir)

        # 1. Extract ALL reference complexes from stored GT Boltz results
        gt_dir = os.path.dirname(ground_truth_ligand_path)
        gt_boltz_dir = os.path.join(gt_dir, "gt_results", "boltz_results_gt_dock")
        ref_complexes = manager.extract_all_complexes_from_results(gt_boltz_dir, "gt_dock")
        ref_pdb_pairs = [(c["lig_pdb"], c["prot_pdb"]) for c in ref_complexes]

        # 2. Run Boltz for Agent (with co-ligands if present)
        protein_sequences = BoltzManager.parse_fasta(protein_fasta_path)
        with open(agent_file_path, 'r') as f:
            agent_smiles = f.read().strip().split()[0].replace('"', '').replace("'", "")

        yaml_path = manager.create_input_yaml(
            protein_sequences,
            agent_smiles,
            job_name="agent_dock",
            pocket_contacts=pocket_contacts,
            predict_affinity=True,
            co_ligands=co_ligands,
        )
        results_dir = manager.run_prediction(yaml_path, diffusion_samples=diffusion_samples)
        agent_boltz_dir = os.path.join(results_dir, "boltz_results_agent_dock")

        # 3. Extract ALL agent complexes
        agent_complexes = manager.extract_all_complexes_from_results(agent_boltz_dir, "agent_dock")
        agent_pdb_pairs = [(c["lig_pdb"], c["prot_pdb"]) for c in agent_complexes]

        # 4. Consensus interaction similarity
        similarity = calculate_consensus_tanimoto(agent_pdb_pairs, ref_pdb_pairs, threshold=0.6)

        # 5. Mean binding probability across all agent samples
        binding_prob = manager.get_mean_affinity(agent_boltz_dir, "agent_dock")

        return {
            "interaction_similarity": similarity,
            "binding_probability": binding_prob
        }
