import os
import csv
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

# Interaction Type Constants
TYPE_DONOR = "Donor"
TYPE_ACCEPTOR = "Acceptor"
TYPE_AROMATIC = "Aromatic"
TYPE_HYDROPHOBIC = "Hydrophobic"
TYPE_CATION = "Cation"
TYPE_ANION = "Anion"

# Compatibility Matrix
# Defines which predicted types can match which ground truth types.
# Keys are sets of compatible types.
COMPATIBILITY = {
    TYPE_DONOR: {TYPE_DONOR},
    TYPE_ACCEPTOR: {TYPE_ACCEPTOR},
    TYPE_AROMATIC: {TYPE_AROMATIC, TYPE_HYDROPHOBIC}, # Aromatics are hydrophobic
    TYPE_HYDROPHOBIC: {TYPE_HYDROPHOBIC, TYPE_AROMATIC}, # Hydrophobic can match aromatic locations
    TYPE_CATION: {TYPE_CATION},
    TYPE_ANION: {TYPE_ANION}
}

def parse_agent_csv(file_path):
    """
    Parses the agent's CSV solution file.
    Expected format: x, y, z, type
    Returns: List of dicts {'coords': np.array, 'type': str}
    """
    points = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return []
            
        # Check first line for header
        start_idx = 0
        header_parts = lines[0].strip().split(',')
        try:
            # If first column is float, it's data
            float(header_parts[0])
        except ValueError:
            # It's a header
            start_idx = 1
            
        reader = csv.reader(lines[start_idx:])
        for row in reader:
            if len(row) < 4: 
                continue
            try:
                x, y, z = float(row[0]), float(row[1]), float(row[2])
                p_type = row[3].strip()
                points.append({
                    'coords': np.array([x, y, z]),
                    'type': p_type
                })
            except ValueError:
                continue # Skip malformed rows
    except Exception as e:
        print(f"Error parsing agent CSV: {e}")
        raise e
    
    return points[:3] # We only evaluate the top 3 predictions

def load_ground_truth(file_path):
    """
    Parses the ground truth JSON file.
    Returns: List of dicts {'coords': np.array, 'types': list of str}
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    points = []
    for p in data['points']:
        # Handle both old format 'type' and new 'types'
        if 'types' in p:
            ptypes = p['types']
        else:
            ptypes = [p['type']]
            
        points.append({
            'coords': np.array(p['coords']),
            'types': ptypes
        })
    return points

def are_types_compatible(pred_type, truth_types):
    """
    Checks if the predicted type is compatible with ANY of the truth types.
    truth_types: list of strings
    """
    pred_type = pred_type.strip().capitalize()
    
    for t_type in truth_types:
        t_type = t_type.strip().capitalize()
        
        # Direct match
        if pred_type == t_type:
            return True
            
        # Compatibility lookup
        allowed = COMPATIBILITY.get(t_type, set())
        if pred_type in allowed:
            return True
            
    return False

def evaluate_3d_pharmacophore(agent_solution_path, ground_truth_path, distance_cutoff=2.5):
    """
    Evaluates the agent's solution against the ground truth.
    Returns: dict with 'score' (float 0-1), 'n_correct', 'n_required', 'n_ground_truth', 'matches'.
    """
    if not os.path.exists(agent_solution_path):
        raise FileNotFoundError(f"Solution file not found: {agent_solution_path}")

    predictions = parse_agent_csv(agent_solution_path)
    ground_truth = load_ground_truth(ground_truth_path)

    if len(predictions) == 0:
        raise ValueError("No valid predictions found in solution file")

    n_pred = len(predictions)
    n_true = len(ground_truth)
    
    # Build Cost Matrix
    # cost_matrix[i, j] = distance if compatible, else large_penalty
    LARGE_PENALTY = 1000.0
    cost_matrix = np.full((n_pred, n_true), LARGE_PENALTY)
    
    for i, pred in enumerate(predictions):
        for j, truth in enumerate(ground_truth):
            if are_types_compatible(pred['type'], truth['types']):
                dist = np.linalg.norm(pred['coords'] - truth['coords'])
                cost_matrix[i, j] = dist
    
    # Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    correct_matches = 0
    matched_details = []
    
    for r, c in zip(row_ind, col_ind):
        dist = cost_matrix[r, c]
        is_match = dist < distance_cutoff
        
        match_info = {
            "pred_idx": int(r),
            "true_idx": int(c),
            "distance": float(dist) if dist < LARGE_PENALTY else "Incompatible",
            "pred_type": predictions[r]['type'],
            "true_types": ground_truth[c]['types'],
            "success": bool(is_match)
        }
        matched_details.append(match_info)
        
        if is_match:
            correct_matches += 1
            
    # Final Score: Fraction of correctly matched points (0/3, 1/3, 2/3, 3/3)
    # The agent is asked for 3 points.
    # Ground truth may have up to 5 points.
    # Score is (matches / 3).
    REQUIRED_POINTS = 3
    score = correct_matches / REQUIRED_POINTS
    if score > 1.0: score = 1.0 # Should not happen if parser caps at 3
    
    return {
        "score": score,
        "matches": matched_details,
        "n_correct": correct_matches,
        "n_required": REQUIRED_POINTS,
        "n_ground_truth": n_true
    }
