from rdkit import Chem
from rdkit.Chem import QED
import os

def is_valid_smiles(smiles):
    """Checks if a SMILES string is valid."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def qed_score(smiles):
    """Calculates the QED score for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return QED.qed(mol)

def assess_smiles_file_validity(file_path):
    """
    Reads a file containing a SMILES string and checks if it is valid.
    Assumes the file contains a single SMILES string on the first line.
    """
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r') as f:
        content = f.read().strip()
        # Handle cases where user might put quotes
        content = content.replace('"', '').replace("'", "")
    
    return is_valid_smiles(content)

def assess_smiles_file_qed(file_path):
    """
    Reads a file containing a SMILES string and returns its QED score.
    Assumes the file contains a single SMILES string on the first line.
    """
    if not os.path.exists(file_path):
        return 0.0
        
    with open(file_path, 'r') as f:
        content = f.read().strip()
        content = content.replace('"', '').replace("'", "")
        
    return qed_score(content)