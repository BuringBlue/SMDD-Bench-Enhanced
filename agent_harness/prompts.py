"""System prompt and tool schemas for the SMDD-Bench agent."""

SYSTEM_PROMPT = """You are a computational medicinal chemist assistant. You are given a drug discovery task and must use your tools to solve it.

Available tools:
- python(code): Execute Python code. Available packages: rdkit, numpy, pandas, scipy, scikit-learn, biopython, biotite, openbabel, plip. To compute synthetic accessibility (SA) scores: `from rdkit.Chem import RDConfig; import sys, os; sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score')); import sascorer; sascorer.calculateScore(mol)`
- predict_boltz(smiles, protein_sequences, pocket_residues, co_ligand_smiles, num_samples): Run Boltz structure prediction. Returns binding probability and affinity. Budget: {boltz_budget} calls.
- predict_admet(smiles): Predict ADMET properties. Returns hERG, BBB, solubility, CYP3A4, clearance, Caco2, PPB, AMES. Budget: {admet_budget} calls.
- submit(file_path): Submit your solution file.

You have no internet access. Work only with the files provided in /workspace. Outline your thought process before making tool calls.
When ready, call submit() with the path to your solution file (relative to /workspace)."""


def build_system_prompt(config):
    return SYSTEM_PROMPT.format(
        boltz_budget=config.boltz_budget,
        admet_budget=config.admet_budget,
    )


TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "python",
        "description": "Execute Python code in the workspace environment. Available packages: rdkit, numpy, pandas, scipy, scikit-learn, biopython, biotite, openbabel, plip. Working directory is /workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Explain your thought process: what are you trying to accomplish with this call and why?",
                },
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                }
            },
            "required": ["reasoning", "code"],
        },
    },
    {
        "type": "function",
        "name": "predict_boltz",
        "description": "Run Boltz co-folding and binding affinity prediction for a small molecule against a protein. Returns binding probability, affinity prediction, and CIF structure files. Each call counts against your Boltz budget.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Explain your thought process: what are you trying to accomplish with this call and why?",
                },
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the ligand to predict",
                },
                "protein_sequences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Chain ID"},
                            "sequence": {"type": "string", "description": "Amino acid sequence"},
                        },
                        "required": ["id", "sequence"],
                    },
                    "description": "Protein chain sequences",
                },
                "pocket_residues": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {},
                    },
                    "description": "Optional list of [chain_id, residue_number] pairs defining the binding pocket",
                },
                "co_ligand_smiles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "smiles": {"type": "string"},
                        },
                        "required": ["id", "smiles"],
                    },
                    "description": "Optional co-ligands for co-folding",
                }
            },
            "required": ["reasoning", "smiles", "protein_sequences"],
        },
    },
    {
        "type": "function",
        "name": "predict_admet",
        "description": "Predict ADMET properties for a molecule using a machine learning model. Returns: hERG, BBB, solubility, CYP3A4, clearance, Caco2, PPB, AMES. Each call counts against your ADMET budget.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Explain your thought process: what are you trying to accomplish with this call and why?",
                },
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule",
                }
            },
            "required": ["reasoning", "smiles"],
        },
    },
    {
        "type": "function",
        "name": "submit",
        "description": "Submit your solution file. This ends the task. The file must exist in /workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Explain your thought process: what are you trying to accomplish with this call and why?",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the solution file relative to /workspace (e.g. 'solution.smi')",
                }
            },
            "required": ["reasoning", "file_path"],
        },
    },
]
