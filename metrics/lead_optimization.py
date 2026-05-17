import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem, DataStructs, RDConfig
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from metrics.boltz_manager import BoltzManager

# SA Score from RDKit contrib
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# Lazy singleton for ADMET-AI model
_admet_model = None

def _get_admet_model():
    global _admet_model
    if _admet_model is None:
        import functools, torch, contextlib, io
        _original_load = torch.load
        torch.load = functools.partial(_original_load, weights_only=False)
        try:
            from admet_ai import ADMETModel
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _admet_model = ADMETModel()
        finally:
            torch.load = _original_load
    return _admet_model

# ADMET-AI dataset name -> our canonical property name
_ADMET_DATASET_TO_PROPERTY = {
    "hERG": "herg",
    "BBB_Martins": "bbb",
    "Solubility_AqSolDB": "solubility",
    "CYP3A4_Veith": "cyp3a4",
    "Clearance_Hepatocyte_AZ": "clearance",
    "Caco2_Wang": "caco2",
    "PPBR_AZ": "ppb",
    "AMES": "ames",
}
_PROPERTY_TO_DATASET = {v: k for k, v in _ADMET_DATASET_TO_PROPERTY.items()}

# PAINS and Brenk filter catalogs (built once)
_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog(_pains_params)

_brenk_params = FilterCatalogParams()
_brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
_BRENK_CATALOG = FilterCatalog(_brenk_params)


def read_smiles_file(file_path):
    """Read a SMILES string from an agent output file."""
    with open(file_path, 'r') as f:
        content = f.read().strip()
    content = content.split()[0].replace('"', '').replace("'", "")
    return content


def assess_hard_constraints(agent_file_path, reference_file_path, similarity_threshold=0.7, skip_tanimoto=False):
    """Check all RDKit-computable hard constraints (excludes Boltz binding probability).

    Args:
        agent_file_path: path to file containing agent's SMILES
        reference_file_path: path to reference ligand SDF, used for tanimoto similarity with agent SMILES.
                             Can be None when skip_tanimoto=True.
        similarity_threshold: minimum Tanimoto similarity to reference (default 0.7)
        skip_tanimoto: if True, skip the Tanimoto similarity check entirely (e.g. for task 5
                       where there is no single reference molecule)

    Returns:
        dict with "all_passed" bool and per-constraint details
    """
    smiles = read_smiles_file(agent_file_path)
    results = {}

    # --- SMILES validity ---
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        results["smiles_valid"] = {"passed": False, "value": None, "reason": "Cannot parse SMILES"}
        results["all_passed"] = False
        return results

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        results["smiles_valid"] = {"passed": False, "value": None, "reason": f"Sanitization failed: {e}"}
        results["all_passed"] = False
        return results

    # Check for radicals
    has_radicals = any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms())
    if has_radicals:
        results["smiles_valid"] = {"passed": False, "value": None, "reason": "Contains radicals"}
        results["all_passed"] = False
        return results

    # Check for disconnected fragments
    if "." in smiles:
        results["smiles_valid"] = {"passed": False, "value": None, "reason": "Disconnected fragments"}
        results["all_passed"] = False
        return results

    results["smiles_valid"] = {"passed": True, "value": smiles}

    # --- Molecular weight < 600 ---
    mw = Descriptors.MolWt(mol)
    results["molecular_weight"] = {"passed": mw < 600, "value": round(mw, 2), "threshold": "< 600"}

    # --- LogP in [-1, 5] ---
    logp = Crippen.MolLogP(mol)
    results["logp"] = {"passed": -1 <= logp <= 5, "value": round(logp, 2), "threshold": "[-1, 5]"}

    # --- TPSA < 140 ---
    tpsa = Descriptors.TPSA(mol)
    results["tpsa"] = {"passed": tpsa < 140, "value": round(tpsa, 2), "threshold": "< 140"}

    # --- HBD <= 5 ---
    hbd = Lipinski.NumHDonors(mol)
    results["hbd"] = {"passed": hbd <= 5, "value": hbd, "threshold": "<= 5"}

    # --- HBA <= 10 ---
    hba = Lipinski.NumHAcceptors(mol)
    results["hba"] = {"passed": hba <= 10, "value": hba, "threshold": "<= 10"}

    # --- Rotatable bonds <= 10 ---
    rot = Lipinski.NumRotatableBonds(mol)
    results["rotatable_bonds"] = {"passed": rot <= 10, "value": rot, "threshold": "<= 10"}

    # --- Net formal charge in [-2, +2] ---
    charge = Chem.GetFormalCharge(mol)
    results["formal_charge"] = {"passed": -2 <= charge <= 2, "value": charge, "threshold": "[-2, +2]"}

    # --- SA score < 4.5 ---
    sa = sascorer.calculateScore(mol)
    results["sa_score"] = {"passed": sa < 4.5, "value": round(sa, 2), "threshold": "< 4.5"}

    # --- PAINS filter ---
    pains_match = _PAINS_CATALOG.HasMatch(mol)
    results["pains"] = {"passed": not pains_match, "value": "match" if pains_match else "clean"}

    # --- Brenk filter ---
    brenk_match = _BRENK_CATALOG.HasMatch(mol)
    results["brenk"] = {"passed": not brenk_match, "value": "match" if brenk_match else "clean"}

    # --- Tanimoto similarity >= threshold to reference ---
    if not skip_tanimoto:
        ref_mol = None
        if reference_file_path.endswith('.sdf'):
            suppl = Chem.SDMolSupplier(reference_file_path)
            ref_mol = next(suppl)
        else:
            with open(reference_file_path, 'r') as f:
                ref_smi = f.read().strip().split()[0]
            ref_mol = Chem.MolFromSmiles(ref_smi)

        if ref_mol is None:
            results["tanimoto_similarity"] = {"passed": False, "value": None, "reason": "Cannot load reference"}
        else:
            fp_agent = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_ref = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(fp_agent, fp_ref)
            results["tanimoto_similarity"] = {"passed": sim >= similarity_threshold, "value": round(sim, 4), "threshold": f">= {similarity_threshold}"}

    results["all_passed"] = all(
        v["passed"] for k, v in results.items() if k != "all_passed"
    )
    return results


def assess_admet_properties(agent_file_path):
    """Run ADMET-AI on the agent's molecule.

    Returns:
        dict of canonical property names to predicted values
    """
    import contextlib, io
    smiles = read_smiles_file(agent_file_path)
    model = _get_admet_model()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        preds = model.predict(smiles=smiles)

    result = {}
    for dataset_name, prop_name in _ADMET_DATASET_TO_PROPERTY.items():
        result[prop_name] = float(preds[dataset_name])
    return result


def assess_optimization_objectives(agent_file_path, baseline_values, objectives):
    """Check whether each optimization objective meets its minimum improvement threshold.

    Args:
        agent_file_path: path to file containing agent's SMILES
        baseline_values: dict of property name -> baseline value
        objectives: list of {"property": str, "direction": "minimize"|"maximize", "threshold": float}

    Returns:
        dict with "all_passed" bool and "details" list
    """
    agent_props = assess_admet_properties(agent_file_path)
    details = []

    for obj in objectives:
        prop = obj["property"]
        direction = obj["direction"]
        threshold = obj["threshold"]

        if prop == "binding_affinity":
            # Handled separately by assess_boltz_task4
            details.append({
                "property": prop, "status": "deferred_to_boltz",
                "passed": True  # placeholder; real check in assess_boltz_task4
            })
            continue

        baseline = baseline_values[prop]
        agent_val = agent_props[prop]

        if direction == "minimize":
            improvement = baseline - agent_val
        else:  # maximize
            improvement = agent_val - baseline

        passed = improvement >= threshold
        details.append({
            "property": prop,
            "direction": direction,
            "baseline": baseline,
            "agent_value": agent_val,
            "improvement": round(improvement, 4),
            "threshold": threshold,
            "passed": passed,
        })

    all_passed = all(d["passed"] for d in details)
    return {"all_passed": all_passed, "details": details}


def assess_hold_constant(agent_file_path, baseline_values, hold_constant):
    """Check whether held-constant ADMET properties stay within tolerance.

    Args:
        agent_file_path: path to file containing agent's SMILES
        baseline_values: dict of property name -> baseline value
        hold_constant: list of {"property": str, "tolerance": float, "direction": "increase"|"decrease"|"change"}

    Returns:
        dict with "all_passed" bool and "details" list
    """
    agent_props = assess_admet_properties(agent_file_path)
    details = []

    for hc in hold_constant:
        prop = hc["property"]
        tolerance = hc["tolerance"]
        direction = hc["direction"]

        if prop == "binding_affinity":
            # Handled separately by assess_boltz_task4
            details.append({
                "property": prop, "status": "deferred_to_boltz",
                "passed": True
            })
            continue

        baseline = baseline_values[prop]
        agent_val = agent_props[prop]
        delta = agent_val - baseline

        if direction == "increase":
            # Property must not increase beyond tolerance
            passed = delta <= tolerance
        elif direction == "decrease":
            # Property must not decrease beyond tolerance
            passed = -delta <= tolerance
        else:  # "change" — bidirectional
            passed = abs(delta) <= tolerance

        details.append({
            "property": prop,
            "direction": direction,
            "baseline": baseline,
            "agent_value": agent_val,
            "delta": round(delta, 4),
            "tolerance": tolerance,
            "passed": passed,
        })

    all_passed = all(d["passed"] for d in details)
    return {"all_passed": all_passed, "details": details}


def assess_boltz_task4(agent_file_path, protein_fasta_path, pocket_contacts, co_ligands,
                       baseline_affinity, affinity_objective=None, affinity_hold_constant=None,
                       diffusion_samples=10):
    """Consolidated Boltz evaluation for task 4.

    Runs Boltz co-fold, extracts binding probability and mean affinity_pred_value.

    Args:
        agent_file_path: path to agent's SMILES file
        protein_fasta_path: path to protein FASTA
        pocket_contacts: list of [chain, seqres_index] pairs
        co_ligands: list of {"id": ..., "smiles": ...} or None
        baseline_affinity: baseline affinity_pred_value from generation
        affinity_objective: dict {"direction": "minimize", "threshold": float} if binding_affinity
                           is an optimization objective, else None
        affinity_hold_constant: dict {"tolerance": float} if binding_affinity is held constant, else None
        diffusion_samples: number of diffusion samples

    Returns:
        dict with binding_probability, affinity_pred_value, binding_passed, affinity_passed
    """
    import tempfile

    smiles = read_smiles_file(agent_file_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = BoltzManager(work_dir=tmp_dir)
        protein_sequences = BoltzManager.parse_fasta(protein_fasta_path)

        yaml_path = manager.create_input_yaml(
            protein_sequences,
            smiles,
            job_name="agent_dock",
            pocket_contacts=pocket_contacts,
            predict_affinity=True,
            co_ligands=co_ligands,
        )
        results_dir = manager.run_prediction(yaml_path, diffusion_samples=diffusion_samples)
        agent_boltz_dir = os.path.join(results_dir, "boltz_results_agent_dock")

        # Binding probability (mean across samples) TODO: the average is already taken, so this is redundant, but that's fine since it leads to the same result
        binding_prob = manager.get_mean_affinity(agent_boltz_dir, "agent_dock",
                                                    key="affinity_probability_binary")

        # Affinity pred value (mean across samples) TODO: the average is already taken, so this is redundant, but that's fine since it leads to the same result
        affinity_val = manager.get_mean_affinity(agent_boltz_dir, "agent_dock",
                                                  key="affinity_pred_value")
        
        # IMPORTANT NOTE: affinity_val refers to log10(IC50) prediction outputted by boltz. Note that IC50 is in micro molar units
        # To go from log10(IC50) to pIC50 in kcal/mol, you do (6 - log10(IC50)) * 1.364. 
        # In this case, the lower the affinity_val, the stronger the binding affinity

    binding_passed = binding_prob > 0.7

    # Affinity optimization check
    affinity_opt_passed = True
    if affinity_objective is not None:
        # direction is always "minimize" for binding affinity (lower = stronger)
        improvement = baseline_affinity - affinity_val
        affinity_opt_passed = improvement >= affinity_objective["threshold"]

    # Affinity hold-constant check
    affinity_hold_passed = True
    if affinity_hold_constant is not None:
        # affinity must not increase by more than tolerance (increase = weaker binding)
        delta = affinity_val - baseline_affinity
        affinity_hold_passed = delta <= affinity_hold_constant["tolerance"]

    affinity_passed = affinity_opt_passed and affinity_hold_passed

    return {
        "binding_probability": round(binding_prob, 4),
        "affinity_pred_value": round(affinity_val, 4),
        "baseline_affinity": baseline_affinity,
        "binding_passed": binding_passed,
        "affinity_passed": affinity_passed,
        "all_passed": binding_passed and affinity_passed,
    }
