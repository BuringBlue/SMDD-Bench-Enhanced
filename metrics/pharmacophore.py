import os
import sys
import importlib.util
from rdkit import Chem

def load_module_from_path(path, module_name):
    """Dynamically loads a python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Could not load specifications for module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def assess_recall(agent_file_path, hidden_actives_path):
    """
    Calculates the recall of the agent's pharmacophore function.
    Recall = (True Positives) / (Total Actives)
    """
    if not os.path.exists(agent_file_path):
        raise FileNotFoundError(f"Agent solution file not found: {agent_file_path}")
    if not os.path.exists(hidden_actives_path):
        raise FileNotFoundError(f"Hidden actives file not found: {hidden_actives_path}")

    # Load agent's code
    try:
        module = load_module_from_path(agent_file_path, "agent_solution_recall")
        if not hasattr(module, 'check_pharmacophore'):
            raise AttributeError("Agent solution must contain a 'check_pharmacophore(smiles)' function.")
        check_func = module.check_pharmacophore
    except Exception as e:
        print(f"Error loading agent code: {e}")
        return 0.0

    # Load actives
    with open(hidden_actives_path, 'r') as f:
        # Assuming one SMILES per line, possibly with ID
        lines = f.readlines()
    
    # Simple parser: take first token as SMILES
    actives = [line.strip().split()[0] for line in lines if line.strip()]
    
    if not actives:
        return 0.0

    hits = 0
    valid_molecules = 0
    for smi in actives:
        # Validate SMILES first to ensure test fairness
        if Chem.MolFromSmiles(smi) is None:
            continue
        valid_molecules += 1
        
        try:
            result = check_func(smi)
            if result is True:
                hits += 1
        except Exception:
            # If function crashes on a molecule, count as False
            pass

    if valid_molecules == 0:
        return 0.0

    return hits / valid_molecules

def assess_specificity(agent_file_path, hidden_decoys_path):
    """
    Calculates the specificity (True Negative Rate) of the agent's pharmacophore function.
    Specificity = (True Negatives) / (Total Decoys)
    """
    if not os.path.exists(agent_file_path):
        raise FileNotFoundError(f"Agent solution file not found: {agent_file_path}")
    if not os.path.exists(hidden_decoys_path):
        raise FileNotFoundError(f"Hidden decoys file not found: {hidden_decoys_path}")

    # Load agent's code
    try:
        module = load_module_from_path(agent_file_path, "agent_solution_spec")
        if not hasattr(module, 'check_pharmacophore'):
            raise AttributeError("Agent solution must contain a 'check_pharmacophore(smiles)' function.")
        check_func = module.check_pharmacophore
    except Exception as e:
        print(f"Error loading agent code: {e}")
        return 0.0

    # Load decoys
    with open(hidden_decoys_path, 'r') as f:
        lines = f.readlines()
    
    decoys = [line.strip().split()[0] for line in lines if line.strip()]
    
    if not decoys:
        return 0.0

    true_negatives = 0
    valid_molecules = 0
    for smi in decoys:
        if Chem.MolFromSmiles(smi) is None:
            continue
        valid_molecules += 1
        
        try:
            result = check_func(smi)
            # We want result to be False for decoys
            if result is False:
                true_negatives += 1
        except Exception:
            # Crash does NOT count as a correct rejection — the function
            # must explicitly return False to demonstrate it can handle
            # the molecule.  Treating exceptions as negatives would let
            # an overly narrow filter game specificity for free.
            pass

    if valid_molecules == 0:
        return 0.0

    return true_negatives / valid_molecules
