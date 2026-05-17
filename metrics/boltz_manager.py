import os
import shutil
import subprocess
import yaml
import json
import urllib.request
from rdkit import Chem
from rdkit.Chem import AllChem

class BoltzManager:
    def __init__(self, work_dir="."):
        self.work_dir = os.path.abspath(work_dir)
        self.boltz_cmd = "boltz"

    @staticmethod
    def _parse_chain_ids(header):
        """Extract auth chain ID(s) from an RCSB FASTA header.

        Handles both singular and plural forms:
            "|Chain A|"              -> ["A"]
            "|Chain B[auth C]|"      -> ["C"]
            "|Chains A, B|"          -> ["A", "B"]
            "|Chains A, B[auth C]|"  -> ["A", "C"]

        Auth chain IDs are preferred because the rest of the pipeline
        (pocket contacts, PLIP, PDB SEQRES) uses auth IDs.
        """
        import re

        for keyword in ("|Chains ", "|Chain "):
            if keyword in header:
                ids_str = header.split(keyword)[1].split("|")[0].strip()
                raw_ids = [c.strip() for c in ids_str.split(",")]
                parsed = []
                for raw in raw_ids:
                    m = re.search(r'\[auth\s+(\S+)\]', raw)
                    if m:
                        parsed.append(m.group(1))
                    else:
                        parsed.append(raw)
                return parsed
        return [header.split()[0]]

    @staticmethod
    def parse_fasta(fasta_path):
        protein_sequences = []
        try:
            with open(fasta_path, 'r') as f:
                lines = f.readlines()

            current_ids = None
            current_seq = []
            for line in lines:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if current_seq and current_ids:
                        seq = "".join(current_seq)
                        for cid in current_ids:
                            protein_sequences.append({"id": cid, "sequence": seq})
                    current_ids = BoltzManager._parse_chain_ids(line[1:])
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_seq and current_ids:
                seq = "".join(current_seq)
                for cid in current_ids:
                    protein_sequences.append({"id": cid, "sequence": seq})

            if not protein_sequences:
                protein_seq = "".join([l.strip() for l in lines])
                if protein_seq:
                    protein_sequences = [{"id": "A", "sequence": protein_seq}]
        except Exception as e:
            print(f"Error reading protein FASTA: {e}")
            return []
        return protein_sequences

    @staticmethod
    def _assign_bond_orders(mol, template_smiles):
        """Reassign bond orders on *mol* using a SMILES template.

        Returns the corrected mol, or the original mol if assignment fails.
        """
        template_mol = Chem.MolFromSmiles(template_smiles)
        if template_mol is None:
            return mol
        try:
            corrected = AllChem.AssignBondOrdersFromTemplate(template_mol, mol)
            return corrected
        except Exception:
            return mol

    @staticmethod
    def fetch_ligand_smiles(ligand_code):
        """Fetch canonical SMILES for a PDB ligand from the RCSB Chemical Component Dictionary.

        Tries the ideal SDF first, then falls back to the CCD REST API.
        Returns the SMILES string or None.
        """
        ligand_code = ligand_code.strip().upper()

        # Try ideal SDF
        sdf_url = f"https://files.rcsb.org/ligands/view/{ligand_code}_ideal.sdf"
        try:
            with urllib.request.urlopen(sdf_url, timeout=10) as resp:
                sdf_block = resp.read().decode("utf-8")
            mol = Chem.MolFromMolBlock(sdf_block, removeHs=True)
            if mol:
                return Chem.MolToSmiles(mol)
        except Exception:
            pass

        # Fallback: CCD REST API (returns JSON with rcsb_chem_comp_descriptor)
        api_url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_code}"
        try:
            with urllib.request.urlopen(api_url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            descriptors = data.get("rcsb_chem_comp_descriptor", [])
            for desc in descriptors:
                if desc.get("type") == "SMILES_CANONICAL":
                    return desc["descriptor"]
                if desc.get("type") == "SMILES":
                    return desc["descriptor"]
        except Exception:
            pass

        return None

    def is_available(self):
        return shutil.which(self.boltz_cmd) is not None

    def create_input_yaml(self, protein_sequences, ligand_smiles, job_name="boltz_run", pocket_contacts=None, predict_affinity=True, co_ligands=None):
        """
        Args:
            protein_sequences: str or list of {"id": ..., "sequence": ...} dicts
            ligand_smiles: SMILES string for the target ligand (assigned id "LIG")
            co_ligands: optional list of {"id": ..., "smiles": ...} dicts for additional ligands
                        to include in co-folding (e.g. [{"id": "LIG2", "smiles": "..."}])
            pocket_contacts: pocket constraint applied only to the target ligand "LIG"
            predict_affinity: affinity prediction applied only to the target ligand "LIG"
        """
        sequences = []
        if isinstance(protein_sequences, str):
            sequences.append({"protein": {"id": "A", "sequence": protein_sequences}})
        else:
            for ps in protein_sequences:
                sequences.append({"protein": ps})

        sequences.append({"ligand": {"id": "LIG", "smiles": ligand_smiles}})

        if co_ligands:
            # Boltz rejects systems with duplicate ligand SMILES when affinity is enabled.
            # Deduplicate co-ligands by canonical SMILES, also dropping any that match the target.
            from rdkit import Chem
            target_canon = Chem.CanonSmiles(ligand_smiles)
            seen_smiles = {target_canon}
            for co_lig in co_ligands:
                canon = Chem.CanonSmiles(co_lig["smiles"])
                if canon in seen_smiles:
                    continue
                seen_smiles.add(canon)
                sequences.append({"ligand": {"id": co_lig["id"], "smiles": co_lig["smiles"]}})

        data = {"version": 1, "sequences": sequences}
        if pocket_contacts:
            data["constraints"] = [{"pocket": {"binder": "LIG", "contacts": pocket_contacts, "max_distance": 6.0, "force": True}}]

        if predict_affinity:
            data["properties"] = [{"affinity": {"binder": "LIG"}}]

        os.makedirs(self.work_dir, exist_ok=True)
        yaml_path = os.path.join(self.work_dir, f"{job_name}.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        return yaml_path

    def run_prediction(self, yaml_path, output_dir_name="boltz_results", diffusion_samples=10):
        output_dir = os.path.join(self.work_dir, output_dir_name)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        cmd = [self.boltz_cmd, "predict", yaml_path, "--out_dir", output_dir, "--use_msa_server", "--use_potentials", "--diffusion_samples", str(diffusion_samples), "--seed", "42", "--diffusion_samples_affinity", str(diffusion_samples), "--max_parallel_samples", "2"]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            # Check if any prediction files were produced despite the error
            import glob as _glob
            prediction_files = _glob.glob(os.path.join(output_dir, "**", "*.cif"), recursive=True)
            if not prediction_files:
                print(f"Boltz prediction failed (exit code {result.returncode}) and produced no output files. Possible OOM or other fatal error.")
                return None
            print(f"Warning: Boltz exited with code {result.returncode} but produced {len(prediction_files)} prediction files. Continuing.")
        return output_dir

    def extract_predict_results(self, results_dir, job_name):
        results_file = os.path.join(results_dir, "predictions", job_name, f"confidence_{job_name}_model_0.json")
        with open(results_file, 'r') as f:
            return json.load(f)

    def extract_affinity_results(self, results_dir, job_name):
        results_file = os.path.join(results_dir, "predictions", job_name, f"affinity_{job_name}.json")
        with open(results_file, 'r') as f:
            return json.load(f)

    def select_best_model(self, results_dir, job_name, key="affinity_probability_binary"):
        """Select the model with highest value for the given affinity key.

        Boltz affinity JSON has ensemble values (unsuffixed) and per-model values
        with 1-indexed suffixes, e.g.:
            "affinity_probability_binary": 0.84,   # ensemble
            "affinity_probability_binary1": 0.0,    # model 1 -> CIF model_0
            "affinity_probability_binary2": 0.84,   # model 2 -> CIF model_1

        Returns (best_model_index, best_value) where model_index is 0-based (for CIF filenames).
        """
        affinity_data = self.extract_affinity_results(results_dir, job_name)

        # Collect per-model values: key1 -> model_0, key2 -> model_1, etc.
        per_model = {}
        for k, v in affinity_data.items():
            if k.startswith(key) and k[len(key):].isdigit():
                suffix = int(k[len(key):])
                per_model[suffix] = float(v)

        if per_model:
            best_suffix = max(per_model, key=per_model.get)
            return best_suffix - 1, per_model[best_suffix]

        # Single sample: no suffixed keys, just the ensemble/only value
        return 0, float(affinity_data[key])

    def select_most_confident_model(self, results_dir, job_name, key="confidence_score"):
        """Select the model with the highest confidence score.

        Reads per-model confidence JSON files:
            predictions/{job_name}/confidence_{job_name}_model_{i}.json

        Returns (best_model_index, best_value) where model_index is 0-based.
        """
        import glob as _glob
        predictions_dir = os.path.join(results_dir, "predictions", job_name)
        conf_files = sorted(_glob.glob(os.path.join(predictions_dir, f"confidence_{job_name}_model_*.json")))

        best_idx, best_val = 0, -1.0
        for conf_file in conf_files:
            idx = int(conf_file.rsplit("_", 1)[1].split(".")[0])
            with open(conf_file, 'r') as f:
                data = json.load(f)
            val = float(data[key])
            if val > best_val:
                best_idx, best_val = idx, val

        return best_idx, best_val

    def extract_ligand_from_cif(self, cif_path, ligand_chain_id="LIG", template_smiles=None):
        """Extract ligand mol from a specific CIF file."""
        import biotite.structure.io.pdbx as pdbx
        import biotite.structure.io.pdb as pdb

        structure = pdbx.get_structure(pdbx.CIFFile.read(cif_path), model=1, use_author_fields=True)
        ligand_struct = structure[structure.chain_id == ligand_chain_id]
        if not len(ligand_struct):
            return None

        ligand_struct.chain_id[:] = 'L'
        ligand_struct.res_name[:] = 'LIG'
        tmp_pdb = pdb.PDBFile()
        tmp_pdb.set_structure(ligand_struct)
        mol = Chem.MolFromPDBBlock("\n".join(tmp_pdb.lines), sanitize=False)
        if mol:
            if template_smiles:
                mol = self._assign_bond_orders(mol, template_smiles) # Assign bond orders using template smiles
            try:
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol, addCoords=True)
            except:
                pass
        return mol

    def extract_ligand_conformers(self, results_dir, ligand_chain_id="LIG", template_smiles=None):
        import biotite.structure.io.pdbx as pdbx
        import biotite.structure.io.pdb as pdb
        conformers = []
        for root, _, files in os.walk(results_dir):
            for f in files:
                if f.endswith((".cif", ".pdb")):
                    file_path = os.path.join(root, f)
                    try:
                        if f.endswith(".cif"):
                            structure = pdbx.get_structure(pdbx.CIFFile.read(file_path), model=1, use_author_fields=True)
                        else:
                            structure = pdb.PDBFile.read(file_path).get_structure(model=1)

                        ligand_struct = structure[structure.chain_id == ligand_chain_id]
                        if not len(ligand_struct):
                            continue

                        ligand_struct.chain_id[:] = 'L'
                        ligand_struct.res_name[:] = 'LIG'
                        tmp_pdb = pdb.PDBFile()
                        tmp_pdb.set_structure(ligand_struct)
                        mol = Chem.MolFromPDBBlock("\n".join(tmp_pdb.lines), sanitize=False)
                        if mol:
                            if template_smiles:
                                mol = self._assign_bond_orders(mol, template_smiles)
                            try:
                                Chem.SanitizeMol(mol)
                                mol = Chem.AddHs(mol, addCoords=True)
                            except: pass
                            conformers.append(mol)
                    except Exception:
                        continue
        return conformers

    def extract_complex_from_cif(self, cif_path, ligand_chain_id="LIG", template_smiles=None):
        """Extract ligand and protein data from a specific CIF file for PLIP analysis."""
        import biotite.structure.io.pdbx as pdbx
        import biotite.structure.io.pdb as pdb

        structure = pdbx.get_structure(pdbx.CIFFile.read(cif_path), model=1, use_author_fields=True)

        ligand_struct = structure[structure.chain_id == ligand_chain_id]
        ligand_struct.chain_id[:] = 'L'
        ligand_struct.res_name[:] = 'LIG'

        tmp_pdb_lig = pdb.PDBFile()
        tmp_pdb_lig.set_structure(ligand_struct)
        lig_pdb_block = "\n".join(tmp_pdb_lig.lines)
        lig_mol = Chem.MolFromPDBBlock(lig_pdb_block, sanitize=False)
        if lig_mol and template_smiles:
            lig_mol = self._assign_bond_orders(lig_mol, template_smiles)

        protein_struct = structure[structure.chain_id != ligand_chain_id]
        # Remap co-ligand chain IDs and residue names for PDB compat
        # PDB format requires chain IDs <= 1 char and residue names <= 3 chars
        co_lig_counter = 0
        for chain_id in set(protein_struct.chain_id):
            if len(chain_id) > 1:
                new_chain = chr(ord('M') + co_lig_counter)
                mask = protein_struct.chain_id == chain_id
                protein_struct.chain_id[mask] = new_chain
                co_lig_counter += 1
        for i, rname in enumerate(protein_struct.res_name):
            if len(rname) > 3:
                protein_struct.res_name[i] = rname[:3]
        tmp_pdb_prot = pdb.PDBFile()
        tmp_pdb_prot.set_structure(protein_struct)
        prot_pdb_block = "\n".join(tmp_pdb_prot.lines)
        prot_mol = Chem.MolFromPDBBlock(prot_pdb_block, sanitize=False)

        if lig_mol and prot_mol:
            try:
                Chem.SanitizeMol(lig_mol)
                Chem.SanitizeMol(prot_mol)
            except:
                pass
            return {
                "lig_mol": lig_mol,
                "prot_mol": prot_mol,
                "lig_pdb": lig_pdb_block,
                "prot_pdb": prot_pdb_block
            }
        return None

    def get_mean_affinity(self, results_dir, job_name, key="affinity_probability_binary"):
        """Return mean of per-model affinity values.

        Boltz affinity JSON has per-model values with 1-indexed suffixes:
            "affinity_probability_binary1": 0.0,    # model 1 -> CIF model_0
            "affinity_probability_binary2": 0.84,   # model 2 -> CIF model_1
        Returns the arithmetic mean of all per-model values.
        Falls back to the unsuffixed ensemble value if no per-model keys exist.
        """
        affinity_data = self.extract_affinity_results(results_dir, job_name)

        per_model_values = []
        for k, v in affinity_data.items():
            if k.startswith(key) and k[len(key):].isdigit():
                per_model_values.append(float(v))

        if per_model_values:
            return sum(per_model_values) / len(per_model_values)

        # Single sample: no suffixed keys
        return float(affinity_data[key])

    def extract_all_complexes_from_results(self, results_dir, job_name, ligand_chain_id="LIG"):
        """Extract complex data from ALL model CIFs in a Boltz results directory.

        Returns list of dicts (each with lig_pdb, prot_pdb, lig_mol, prot_mol).
        """
        import glob as _glob
        predictions_dir = os.path.join(results_dir, "predictions", job_name)
        cif_files = sorted(_glob.glob(os.path.join(predictions_dir, f"{job_name}_model_*.cif")))

        complexes = []
        for cif_path in cif_files:
            result = self.extract_complex_from_cif(cif_path, ligand_chain_id)
            if result:
                complexes.append(result)
        return complexes

    def extract_complex_mols(self, results_dir, ligand_chain_id="LIG"):
        import biotite.structure.io.pdbx as pdbx
        import biotite.structure.io.pdb as pdb
        complexes = []

        for root, _, files in os.walk(results_dir):
            for f in files:
                if f.endswith((".cif", ".pdb")):
                    file_path = os.path.join(root, f)
                    result = self.extract_complex_from_cif(file_path, ligand_chain_id)
                    if result:
                        complexes.append(result)

        return complexes