import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem import AllChem, rdFMCS

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

def assess_membership(agent_file_path, ground_truth_file_path):
    if not os.path.exists(agent_file_path):
        return 0.0
    
    if not os.path.exists(ground_truth_file_path):
        return 0.0

    with open(agent_file_path, 'r') as f:
        agent_smiles = f.read().strip().split()[0].replace('"', '').replace("'", "")
    
    agent_mol = Chem.MolFromSmiles(agent_smiles)
    if agent_mol is None:
        return 0.0
    agent_canon = Chem.MolToSmiles(agent_mol)

    valid_answers = set()
    with open(ground_truth_file_path, 'r') as f:
        for line in f:
            smi = line.strip().split()[0]
            m = Chem.MolFromSmiles(smi)
            if m:
                valid_answers.add(Chem.MolToSmiles(m))
    
    return 1.0 if agent_canon in valid_answers else 0.0

def assess_tanimoto_similarity(agent_file_path, reference_file_path):
    """Compute full-molecule Tanimoto similarity (Morgan r=2, 2048 bits)."""
    if not os.path.exists(agent_file_path) or not os.path.exists(reference_file_path):
        return 0.0

    with open(agent_file_path, 'r') as f:
        content = f.read().strip()
        if not content: return 0.0
        agent_smiles = content.split()[0].replace('"', '').replace("'", "")

    agent_mol = Chem.MolFromSmiles(agent_smiles)
    if not agent_mol: return 0.0

    ref_mol = None
    if reference_file_path.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(reference_file_path)
        try:
            ref_mol = next(suppl)
        except:
            pass
    else:
        with open(reference_file_path, 'r') as f:
            ref_smiles = f.read().strip().split()[0]
        ref_mol = Chem.MolFromSmiles(ref_smiles)

    if not ref_mol: return 0.0

    fp_agent = AllChem.GetMorganFingerprintAsBitVect(agent_mol, 2, 2048)
    fp_ref = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, 2048)
    return DataStructs.TanimotoSimilarity(fp_agent, fp_ref)


def assess_scaffold_difference(agent_file_path, reference_file_path):
    if not os.path.exists(agent_file_path) or not os.path.exists(reference_file_path):
        return 0.0
        
    with open(agent_file_path, 'r') as f:
        try:
            content = f.read().strip()
            if not content: return 0.0
            agent_smiles = content.split()[0].replace('"', '').replace("'", "")
        except:
            return 0.0
            
    agent_mol = Chem.MolFromSmiles(agent_smiles)
    if not agent_mol: return 0.0
    
    ref_mol = None
    if reference_file_path.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(reference_file_path)
        try:
            ref_mol = next(suppl)
        except:
            pass
    else:
        with open(reference_file_path, 'r') as f:
            ref_smiles = f.read().strip().split()[0]
        ref_mol = Chem.MolFromSmiles(ref_smiles)
        
    if not ref_mol: return 0.0

    try:
        scaff_agent = MurckoScaffold.GetScaffoldForMol(agent_mol)
        scaff_ref = MurckoScaffold.GetScaffoldForMol(ref_mol)
        fp_agent = AllChem.GetMorganFingerprintAsBitVect(scaff_agent, 2, 2048)
        fp_ref = AllChem.GetMorganFingerprintAsBitVect(scaff_ref, 2, 2048)
        sim = DataStructs.TanimotoSimilarity(fp_agent, fp_ref)
        return 1.0 - sim
    except:
        return 0.0


def assess_scaffold_mcs(agent_file_path, reference_file_path):
    """Compute MCS fraction of Bemis-Murcko scaffolds: MCS_atoms / max(agent_HA, ref_HA).

    Lower values mean more different scaffolds. Returns 1.0 on failure (worst case).
    """
    if not os.path.exists(agent_file_path) or not os.path.exists(reference_file_path):
        return 1.0

    with open(agent_file_path, 'r') as f:
        content = f.read().strip()
        if not content:
            return 1.0
        agent_smiles = content.split()[0].replace('"', '').replace("'", "")

    agent_mol = Chem.MolFromSmiles(agent_smiles)
    if not agent_mol:
        return 1.0

    ref_mol = None
    if reference_file_path.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(reference_file_path)
        try:
            ref_mol = next(suppl)
        except:
            pass
    else:
        with open(reference_file_path, 'r') as f:
            ref_smiles = f.read().strip().split()[0]
        ref_mol = Chem.MolFromSmiles(ref_smiles)

    if not ref_mol:
        return 1.0

    try:
        scaff_agent = MurckoScaffold.GetScaffoldForMol(agent_mol)
        scaff_ref = MurckoScaffold.GetScaffoldForMol(ref_mol)
        agent_ha = scaff_agent.GetNumHeavyAtoms()
        ref_ha = scaff_ref.GetNumHeavyAtoms()
        denom = max(agent_ha, ref_ha)
        if denom == 0:
            return 1.0
        mcs = rdFMCS.FindMCS([scaff_ref, scaff_agent], timeout=10)
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        mcs_atoms = mcs_mol.GetNumAtoms() if mcs_mol else 0
        return mcs_atoms / denom
    except:
        return 1.0