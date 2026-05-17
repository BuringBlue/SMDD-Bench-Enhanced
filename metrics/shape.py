import os
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def _load_mol_from_file(path, add_hs=True, remove_hs=False):
    mol = None
    try:
        if path.endswith('.smi') or path.endswith('.smiles'):
            with open(path, 'r') as f:
                content = f.read().strip().split('\n')[0]
                smiles = content.split()[0].replace('"', '').replace("'", "")
            mol = Chem.MolFromSmiles(smiles)
        elif path.endswith('.sdf'):
            suppl = Chem.SDMolSupplier(path, removeHs=remove_hs)
            mol = next(suppl)
        elif path.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(path, removeHs=remove_hs)
        else:
            with open(path, 'r') as f:
                line = f.readline()
            if "V2000" in line or "V3000" in line or "$$$$" in line:
                suppl = Chem.SDMolSupplier(path, removeHs=remove_hs)
                mol = next(suppl)
            else:
                mol = Chem.MolFromSmiles(line.strip().split()[0])
    except:
        return None

    if mol and add_hs and not remove_hs:
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except:
            pass
    return mol

def _prepare_best_overlay(agent_file_path, reference_file_path):
    """
    Generates conformers for agent, aligns to reference.
    Returns: (best_agent_mol, normalized_score)
    """
    if not os.path.exists(agent_file_path) or not os.path.exists(reference_file_path):
        return None, 0.0

    # 1. Load Molecules
    agent_mol = _load_mol_from_file(agent_file_path, add_hs=True)
    if agent_mol is None: return None, 0.0

    ref_mol = _load_mol_from_file(reference_file_path, add_hs=True)
    if ref_mol is None: return None, 0.0

    # 2. Calculate Reference Self-Score (Volume A)
    try:
        ref_self_o3a = rdMolAlign.GetO3A(ref_mol, ref_mol, prbCid=0, refCid=0)
        score_ref_ref = ref_self_o3a.Score()
        if score_ref_ref == 0: return None, 0.0
    except:
        return None, 0.0

    # 3. Generate Agent Conformers
    try:
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        cids = AllChem.EmbedMultipleConfs(agent_mol, numConfs=50, params=params)
        if not cids: return None, 0.0
        AllChem.MMFFOptimizeMoleculeConfs(agent_mol, numThreads=0)
    except:
        return None, 0.0

    # 4. Calculate Agent Self-Score (Volume B)
    # Since all conformers are roughly same size/features, we pick the first one.
    try:
        agent_self_o3a = rdMolAlign.GetO3A(agent_mol, agent_mol, prbCid=cids[0], refCid=cids[0])
        score_agent_agent = agent_self_o3a.Score()
    except:
        score_agent_agent = score_ref_ref 

    best_raw_score = -1.0
    best_conf_id = -1
    
    # 5. Find Best Alignment
    for cid in cids:
        try:
            o3a = rdMolAlign.GetO3A(agent_mol, ref_mol, prbCid=cid, refCid=0)
            if o3a:
                score = o3a.Score()
                if score > best_raw_score:
                    best_raw_score = score
                    best_conf_id = cid
        except:
            continue
                
    if best_conf_id != -1:
        o3a = rdMolAlign.GetO3A(agent_mol, ref_mol, prbCid=best_conf_id, refCid=0)
        o3a.Align()
        
        best_mol = Chem.Mol(agent_mol)
        best_mol.RemoveAllConformers()
        best_mol.AddConformer(agent_mol.GetConformer(best_conf_id))
        
        # True Tanimoto Normalization: Intersection / Union
        # Union = VolA + VolB - Intersection
        union_score = score_ref_ref + score_agent_agent - best_raw_score
        if union_score <= 0: return None, 0.0
        
        normalized_score = best_raw_score / union_score
        return best_mol, normalized_score

    return None, 0.0

def assess_shape_match(agent_file_path, reference_file_path):
    _, score = _prepare_best_overlay(agent_file_path, reference_file_path)
    return score

def assess_clash(agent_file_path, reference_file_path, protein_file_path, threshold=1.2):
    if not os.path.exists(protein_file_path): return 0.0
    
    agent_mol, _ = _prepare_best_overlay(agent_file_path, reference_file_path)
    if agent_mol is None: return 0.0

    try:
        protein_mol = _load_mol_from_file(protein_file_path, add_hs=False, remove_hs=True)
        if not protein_mol: return 0.0
        
        agent_conf = agent_mol.GetConformer()
        prot_conf = protein_mol.GetConformer()
        
        prot_coords = []
        for i in range(protein_mol.GetNumAtoms()):
            if protein_mol.GetAtomWithIdx(i).GetAtomicNum() > 1:
                pos = prot_conf.GetAtomPosition(i)
                prot_coords.append((pos.x, pos.y, pos.z))
                
        thresh_sq = threshold**2
        
        for i in range(agent_mol.GetNumAtoms()):
            if agent_mol.GetAtomWithIdx(i).GetAtomicNum() <= 1: continue 
            pos = agent_conf.GetAtomPosition(i)
            ax, ay, az = pos.x, pos.y, pos.z
            
            for (px, py, pz) in prot_coords:
                dist_sq = (ax-px)**2 + (ay-py)**2 + (az-pz)**2
                if dist_sq < thresh_sq:
                    return 0.0
                    
        return 1.0 
    except:
        return 0.0