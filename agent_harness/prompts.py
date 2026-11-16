"""System prompt and tool schemas for the SMDD-Bench agent."""

SYSTEM_PROMPT = """You are a computational medicinal chemist assistant. You are given a drug discovery task and must use your tools to solve it.

Available tools:
- python(code): Execute Python code. Available packages: rdkit, numpy, pandas, scipy, scikit-learn, biopython, biotite, openbabel, plip. To compute synthetic accessibility (SA) scores: `from rdkit.Chem import RDConfig; import sys, os; sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score')); import sascorer; sascorer.calculateScore(mol)`
- validate_molecule(smiles): Deterministically validate a molecule with RDKit, including descriptors, SA, reference similarity, PAINS, BRENK, and NIH alerts.
- predict_boltz(smiles, protein_sequences, pocket_residues, co_ligand_smiles, num_samples): Run Boltz structure prediction. Returns binding probability and affinity. Budget: {boltz_budget} calls; the same ligand/target request can execute at most {boltz_max_repeats_per_request} times.
- predict_admet(smiles): Predict ADMET properties. Returns hERG, BBB, solubility, CYP3A4, clearance, Caco2, PPB, AMES. Budget: {admet_budget} calls.
- dock_vina(smiles, receptor_path, center, box_size, search_depth, num_poses): Dock with Vina-GPU 2.1 on an NVIDIA GPU. Returns affinities, pose RMSDs, and pose files. Budget: {vina_budget} calls.
- dock_diffdock(smiles, receptor_path, num_samples, inference_steps): Dock with DiffDock-L 1.1.3. Returns ranked SDF poses and confidence scores. Budget: {diffdock_budget} calls.
- submit(file_path): Submit your solution file.

You have no internet access. Work only with the files in your current workspace. Outline your thought process before making tool calls.
When ready, call submit() with the path to your solution file, relative to the workspace."""


def build_system_prompt(config):
    return SYSTEM_PROMPT.format(
        boltz_budget=config.boltz_budget,
        boltz_max_repeats_per_request=config.boltz_max_repeats_per_request,
        admet_budget=config.admet_budget,
        vina_budget=config.vina_budget,
        diffdock_budget=config.diffdock_budget,
    )


TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "python",
        "description": "Execute Python code in the workspace. Available packages: rdkit, numpy, pandas, scipy, scikit-learn, biopython, biotite, openbabel, plip. Relative paths resolve inside the workspace.",
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
        "name": "validate_molecule",
        "description": "Deterministically validate a candidate with RDKit. Returns parse/sanitization status, fragments, radicals, molecular descriptors, synthetic accessibility, Morgan Tanimoto similarity to reference.sdf, and official RDKit PAINS, BRENK, and NIH filter hits.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Explain why this candidate should be validated.",
                },
                "smiles": {
                    "type": "string",
                    "description": "Candidate SMILES string to validate.",
                },
            },
            "required": ["reasoning", "smiles"],
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
                },
                "num_samples": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Number of Boltz diffusion and affinity samples. Use 1 for smoke tests; larger values improve production predictions at higher cost.",
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
        "name": "dock_vina",
        "description": "Dock a SMILES ligand against a receptor using Vina-GPU 2.1. receptor_path must be a PDB or pre-prepared PDBQT file in the workspace. center and box_size define the search box in Angstrom. Returns affinities, pose RMSDs, and PDBQT/SDF pose paths. Each call counts against the Vina budget.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Explain what this docking call is intended to test.",
                },
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the ligand",
                },
                "receptor_path": {
                    "type": "string",
                    "description": "Path to a receptor .pdb or .pdbqt file relative to the workspace",
                },
                "center": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Docking box center [x, y, z] in Angstrom",
                },
                "box_size": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 1, "maximum": 48},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Docking box dimensions [x, y, z] in Angstrom; defaults to [20, 20, 20]",
                },
                "search_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 128,
                    "description": "Vina-GPU Monte Carlo search depth; defaults to 8",
                },
                "num_poses": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of poses; defaults to 9",
                },
            },
            "required": ["reasoning", "smiles", "receptor_path", "center"],
        },
    },
    {
        "type": "function",
        "name": "dock_diffdock",
        "description": "Dock a SMILES ligand to a receptor PDB using DiffDock-L 1.1.3. Returns ranked SDF poses and model confidence scores; confidence is not binding affinity. Each call counts against the DiffDock budget.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Explain what this docking call is intended to test.",
                },
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the ligand",
                },
                "receptor_path": {
                    "type": "string",
                    "description": "Path to a receptor .pdb file relative to the workspace",
                },
                "num_samples": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 40,
                    "description": "Number of generated poses; defaults to 10",
                },
                "inference_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 40,
                    "description": "Number of diffusion denoising steps; defaults to 20",
                },
            },
            "required": ["reasoning", "smiles", "receptor_path"],
        },
    },
    {
        "type": "function",
        "name": "submit",
        "description": "Submit your solution file. This ends the task. The path must be relative to the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Explain your thought process: what are you trying to accomplish with this call and why?",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the solution file relative to the workspace (e.g. 'solution.smi')",
                }
            },
            "required": ["reasoning", "file_path"],
        },
    },
]
