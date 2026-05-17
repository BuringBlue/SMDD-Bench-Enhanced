import os
import math
import csv

def parse_pharmacophore_file(file_path):
    """
    Parses a CSV file containing pharmacophore points.
    Expected format: FeatureType, X, Y, Z
    Lines starting with # are ignored.
    Returns a list of dicts: [{'type': str, 'x': float, 'y': float, 'z': float}, ...]
    """
    points = []
    valid_types = {
        'donor', 'acceptor', 'hydrophobic', 'aromatic', 'positive', 'negative'
    }
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith('#'):
                    continue
                if len(row) < 4:
                    continue
                
                feat_type = row[0].strip().lower()
                # Normalize types
                if 'cation' in feat_type: feat_type = 'positive'
                if 'anion' in feat_type: feat_type = 'negative'
                if 'h-bond donor' in feat_type: feat_type = 'donor'
                if 'h-bond acceptor' in feat_type: feat_type = 'acceptor'
                
                if feat_type not in valid_types:
                    # Allow fuzzy matching or unknown types? Better to be strict for benchmark.
                    # But for now, let's log and skip or error.
                    continue

                try:
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    points.append({'type': feat_type, 'x': x, 'y': y, 'z': z})
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return []

    return points

def euclidean_distance(p1, p2):
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)

def assess_3d_pharmacophore_overlap(agent_file_path, ground_truth_path, threshold=2.0):
    """
    Evaluates if the agent's predicted points match the ground truth points.
    
    Args:
        agent_file_path: Path to agent's output CSV.
        ground_truth_path: Path to ground truth CSV.
        threshold: Distance threshold in Angstroms (default 2.0).
        
    Returns:
        float: Score between 0.0 and 1.0. 
               (Number of matched Agent Points) / (Number of Expected Agent Points)
               
    Note: The task requires finding *the* 3 conserved points. 
    So we expect the agent to output 3 points.
    The ground truth might contain more than 3 valid conserved points (if the pocket is rich),
    but usually for this task we want them to find the top 3.
    
    If Agent outputs < 3 points: Max score is len(agent_points)/3.
    If Agent outputs > 3 points: We only evaluate the first 3? Or penalize?
    Stricter: Agent MUST output exactly 3 points.
    """
    
    agent_points = parse_pharmacophore_file(agent_file_path)
    gt_points = parse_pharmacophore_file(ground_truth_path)
    
    if not agent_points:
        print("No valid points found in agent solution.")
        return 0.0
    
    if not gt_points:
        print("No ground truth points found. Task configuration error.")
        return 0.0

    # Filter by types first
    matches = 0
    
    matched_gt_indices = set()
    
    for ap in agent_points:
        best_dist = float('inf')
        best_gt_idx = -1
        
        for i, gp in enumerate(gt_points):
            if i in matched_gt_indices:
                continue
            
            if ap['type'] == gp['type']:
                dist = euclidean_distance(ap, gp)
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = i
        
        if best_gt_idx != -1 and best_dist <= float(threshold):
            matches += 1
            matched_gt_indices.add(best_gt_idx)
            
    # Calculate score using F1 to penalize spamming
    if not agent_points:
        return 0.0
        
    precision = matches / len(agent_points)
    recall = matches / len(gt_points)
    
    if precision + recall == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1
