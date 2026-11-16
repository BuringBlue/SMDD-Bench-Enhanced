"""Tool definitions and execution for the SMDD-Bench agent.

Tools execute inside a Docker container via `docker exec`.
Budget tracking for Boltz, ADMET, Vina, and DiffDock is managed host-side.
"""

import json
import hashlib
import os
import posixpath
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from agent_harness.config import AgentConfig


OUTPUT_TRUNCATION_LIMIT = 10_000


class ToolBudgetExceeded(Exception):
    pass


class ToolExecutor:
    def __init__(self, container_id, config: AgentConfig, workspace_dir="/workspace"):
        self.container_id = container_id
        self.config = config
        self.workspace_dir = workspace_dir
        self.boltz_calls = 0
        self.admet_calls = 0
        self.vina_calls = 0
        self.diffdock_calls = 0
        self._boltz_request_calls = {}

    def _docker_exec(self, cmd, timeout=120, workdir=None):
        """Run a command inside the container. Returns (stdout+stderr, returncode)."""
        full_cmd = [
            "docker", "exec", "--workdir", workdir or self.workspace_dir,
            self.container_id,
        ] + cmd
        proc = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if proc.stdout:
            output += proc.stdout
        if proc.stderr:
            output += proc.stderr
        return output, proc.returncode

    def _docker_exec_shell(self, shell_cmd, timeout=120, workdir=None):
        """Run a shell command inside the container. Returns (stdout+stderr, returncode)."""
        full_cmd = [
            "docker", "exec", "--workdir", workdir or self.workspace_dir,
            self.container_id, "bash", "-c", shell_cmd,
        ]
        proc = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if proc.stdout:
            output += proc.stdout
        if proc.stderr:
            output += proc.stderr
        return output, proc.returncode

    def _truncate(self, text):
        if len(text) > OUTPUT_TRUNCATION_LIMIT:
            half = OUTPUT_TRUNCATION_LIMIT // 2
            return (
                text[:half]
                + f"\n\n... [{len(text)} chars total, middle truncated] ...\n\n"
                + text[-half:]
            )
        return text

    def _copy_to_container(self, local_path, container_path):
        """Copy a file from host to container."""
        subprocess.run(
            ["docker", "cp", local_path, f"{self.container_id}:{container_path}"],
            check=True,
        )

    def _copy_from_container(self, container_path, local_path):
        """Copy a file from container to host."""
        subprocess.run(
            ["docker", "cp", f"{self.container_id}:{container_path}", local_path],
            check=True,
        )

    def _prepare_boltz_input(self, boltz_input, call_number):
        """Backend hook for attaching precomputed MSAs."""
        return boltz_input

    def _use_msa_server(self):
        return True

    def _boltz_num_workers(self):
        return 2

    def _exec_boltz(self, command, timeout):
        return self._docker_exec(command, timeout)

    def _exec_vina(self, command, timeout):
        """Backend hook for running the fixed Vina-GPU command with GPU access."""
        return self._docker_exec(command, timeout)

    def _exec_diffdock(self, command, timeout):
        """Backend hook for running the fixed DiffDock command with GPU access."""
        return self._docker_exec(command, timeout)

    def _resolve_workspace_path(self, value, allowed_suffixes):
        """Return a safe backend path and its workspace-relative form."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("workspace path must be a non-empty string")
        value = value.strip()
        workspace = self.workspace_dir.rstrip("/")
        if value.startswith(workspace + "/"):
            relative = value[len(workspace) + 1:]
        elif value.startswith("/workspace/"):
            # The prompt exposes /workspace for both Docker and local backends.
            relative = value[len("/workspace/"):]
        elif value.startswith("/"):
            raise ValueError("path must be inside the workspace")
        else:
            relative = value
        relative = posixpath.normpath(relative)
        if relative in {"", ".", ".."} or relative.startswith("../"):
            raise ValueError("path traversal outside the workspace is not allowed")
        if allowed_suffixes and not relative.lower().endswith(tuple(allowed_suffixes)):
            raise ValueError(f"path must end with one of: {', '.join(allowed_suffixes)}")
        return f"{workspace}/{relative}", relative

    def _read_json_from_container(self, container_path):
        output, returncode = self._docker_exec(["cat", container_path], timeout=30)
        if returncode != 0:
            raise RuntimeError(f"Could not read tool result {container_path}: {output.strip()}")
        return json.loads(output)

    def _make_workspace_paths_relative(self, value):
        """Avoid exposing backend-absolute paths in tool results."""
        workspace_prefix = self.workspace_dir.rstrip("/") + "/"
        if isinstance(value, dict):
            return {key: self._make_workspace_paths_relative(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._make_workspace_paths_relative(item) for item in value]
        if isinstance(value, str) and value.startswith(workspace_prefix):
            return value[len(workspace_prefix):]
        return value

    def dispatch(self, tool_name, arguments):
        """Dispatch a tool call. Returns (result_str, extra_log_data)."""
        counters_before = (
            self.boltz_calls,
            self.admet_calls,
            self.vina_calls,
            self.diffdock_calls,
        )
        boltz_requests_before = self._boltz_request_calls.copy()
        try:
            if tool_name == "python":
                return self.run_python(arguments)
            elif tool_name == "validate_molecule":
                return self.run_validate_molecule(arguments)
            elif tool_name == "predict_boltz":
                return self.run_predict_boltz(arguments)
            elif tool_name == "predict_admet":
                return self.run_predict_admet(arguments)
            elif tool_name == "dock_vina":
                return self.run_dock_vina(arguments)
            elif tool_name == "dock_diffdock":
                return self.run_dock_diffdock(arguments)
            elif tool_name == "submit":
                return self.run_submit(arguments)
            else:
                return f"Unknown tool: {tool_name}", {}
        except Exception as e:
            # Roll back budget and repeat tracking if the call never completed.
            (self.boltz_calls, self.admet_calls,
             self.vina_calls, self.diffdock_calls) = counters_before
            self._boltz_request_calls = boltz_requests_before

            budget_msg = ""
            if tool_name == "predict_boltz":
                budget_msg = f" You have {self.config.boltz_budget - self.boltz_calls} Boltz calls remaining."
            elif tool_name == "predict_admet":
                budget_msg = f" You have {self.config.admet_budget - self.admet_calls} ADMET calls remaining."
            elif tool_name == "dock_vina":
                budget_msg = f" You have {self.config.vina_budget - self.vina_calls} Vina calls remaining."
            elif tool_name == "dock_diffdock":
                budget_msg = f" You have {self.config.diffdock_budget - self.diffdock_calls} DiffDock calls remaining."

            return f"ERROR: Tool '{tool_name}' failed: {type(e).__name__}: {e}.{budget_msg}", {}

    def run_python(self, arguments):
        """Execute Python code inside the container."""
        code = arguments.get("code", "")
        script_name = f".tmp_script_{uuid.uuid4().hex[:8]}.py"
        container_script_path = f"{self.workspace_dir}/{script_name}"

        # Write script to a temp file, copy to container
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            self._copy_to_container(tmp_path, container_script_path)
            output, returncode = self._docker_exec(
                ["python", container_script_path],
                timeout=self.config.python_timeout,
            )
            # Clean up script in container
            self._docker_exec(["rm", "-f", container_script_path], timeout=10)
        except subprocess.TimeoutExpired:
            output = f"ERROR: Python execution timed out after {self.config.python_timeout} seconds."
            returncode = -1
        finally:
            os.unlink(tmp_path)

        return self._truncate(output), {}

    def run_validate_molecule(self, arguments):
        """Run the fixed RDKit validation pipeline inside the task sandbox."""
        smiles = str(arguments.get("smiles", "")).strip()
        if not smiles:
            return "ERROR: validate_molecule requires a non-empty SMILES string.", {}

        source = Path(__file__).resolve().parent.parent / "smdd_runtime" / "validate_molecule.py"
        script_name = f".tmp_validate_{uuid.uuid4().hex[:8]}.py"
        container_script = f"{self.workspace_dir}/{script_name}"
        try:
            self._copy_to_container(str(source), container_script)
            output, returncode = self._docker_exec(
                [
                    "python",
                    container_script,
                    "--smiles",
                    smiles,
                    "--reference",
                    "reference.sdf",
                ],
                timeout=self.config.python_timeout,
            )
        except subprocess.TimeoutExpired:
            return (
                f"ERROR: Molecule validation timed out after {self.config.python_timeout} seconds.",
                {},
            )
        finally:
            try:
                self._docker_exec(["rm", "-f", container_script], timeout=10)
            except Exception:
                pass

        if returncode != 0:
            return f"ERROR: Molecule validation failed (exit code {returncode}).\n{self._truncate(output)}", {}
        return self._truncate(output), {}

    @staticmethod
    def _canonical_smiles(smiles):
        value = str(smiles or "").strip()
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(value)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except Exception:
            pass
        return value

    def _boltz_request_signature(self, arguments):
        proteins = [
            {
                "id": str(item.get("id", "")),
                "sequence": "".join(str(item.get("sequence", "")).split()).upper(),
            }
            for item in arguments.get("protein_sequences", [])
        ]
        co_ligands = [
            {
                "id": str(item.get("id", "")),
                "smiles": self._canonical_smiles(item.get("smiles", "")),
            }
            for item in arguments.get("co_ligand_smiles") or []
        ]
        payload = {
            "smiles": self._canonical_smiles(arguments.get("smiles", "")),
            "protein_sequences": proteins,
            "pocket_residues": arguments.get("pocket_residues") or [],
            "co_ligand_smiles": co_ligands,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def run_predict_boltz(self, arguments):
        """Run Boltz structure prediction inside the container."""
        if self.boltz_calls >= self.config.boltz_budget:
            return (
                f"ERROR: Boltz budget exceeded ({self.config.boltz_budget} calls). "
                "No more Boltz predictions available.",
                {"budget_exceeded": True},
            )

        request_signature = self._boltz_request_signature(arguments)
        request_calls = self._boltz_request_calls.get(request_signature, 0)
        repeat_limit = self.config.boltz_max_repeats_per_request
        if request_calls >= repeat_limit:
            return (
                "ERROR: Repeated Boltz request limit reached. "
                f"This ligand/target request has already run {request_calls} times "
                f"(limit: {repeat_limit}). Use the existing predictions, test a "
                "different molecule/target request, or submit the best validated candidate.",
                {
                    "repeat_limit_exceeded": True,
                    "request_calls": request_calls,
                    "budget_remaining": self.config.boltz_budget - self.boltz_calls,
                },
            )

        self.boltz_calls += 1
        self._boltz_request_calls[request_signature] = request_calls + 1
        call_number = self.boltz_calls

        smiles = arguments["smiles"]
        protein_sequences = arguments["protein_sequences"]
        pocket_residues = arguments.get("pocket_residues")
        co_ligand_smiles = arguments.get("co_ligand_smiles")
        num_samples = arguments.get("num_samples", 10)

        # Build Boltz input YAML (reusing BoltzManager logic)
        sequences = []
        for ps in protein_sequences:
            sequences.append({"protein": {"id": ps["id"], "sequence": ps["sequence"]}})
        sequences.append({"ligand": {"id": "LIG", "smiles": smiles}})

        if co_ligand_smiles:
            for co_lig in co_ligand_smiles:
                sequences.append({"ligand": {"id": co_lig["id"], "smiles": co_lig["smiles"]}})

        import yaml
        boltz_input = {"version": 1, "sequences": sequences}
        if pocket_residues:
            boltz_input["constraints"] = [{
                "pocket": {
                    "binder": "LIG",
                    "contacts": pocket_residues,
                    "max_distance": 6.0,
                    "force": True,
                }
            }]
        boltz_input["properties"] = [{"affinity": {"binder": "LIG"}}]
        boltz_input = self._prepare_boltz_input(boltz_input, call_number)

        # Write YAML to temp file, copy to container
        job_name = f"boltz_call_{call_number}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(boltz_input, f, sort_keys=False)
            tmp_yaml = f.name

        container_yaml = f"{self.workspace_dir}/{job_name}.yaml"
        container_out_dir = f"{self.workspace_dir}/boltz_results/call_{call_number}"
        self._copy_to_container(tmp_yaml, container_yaml)
        os.unlink(tmp_yaml)

        # Run Boltz inside container
        boltz_cmd = [
            "/opt/oracle/bin/boltz", "predict", container_yaml,
            "--out_dir", container_out_dir,
            "--use_potentials",
            "--diffusion_samples", str(num_samples),
            "--seed", str(self.config.seed),
            "--diffusion_samples_affinity", str(num_samples),
            "--max_parallel_samples", "2",
            "--num_workers", str(self._boltz_num_workers()),
        ]
        if self._use_msa_server():
            boltz_cmd.append("--use_msa_server")

        try:
            output, returncode = self._exec_boltz(
                boltz_cmd,
                timeout=self.config.boltz_timeout,
            )
        except subprocess.TimeoutExpired:
            return (
                f"ERROR: Boltz prediction timed out after {self.config.boltz_timeout} seconds.",
                {"call_number": call_number, "budget_remaining": self.config.boltz_budget - self.boltz_calls},
            )

        if returncode != 0:
            return (
                f"Boltz prediction may have encountered issues (exit code {returncode}).\n"
                f"Output directory: boltz_results/call_{call_number}\n"
                f"Output:\n{self._truncate(output)}",
                {"call_number": call_number, "budget_remaining": self.config.boltz_budget - self.boltz_calls},
            )

        # Try to extract affinity results
        # Boltz puts output under {out_dir}/boltz_results_{job_name}/
        boltz_results_subdir = f"{container_out_dir}/boltz_results_{job_name}"
        affinity_script = f"""
import json, glob, os
results_dir = "{boltz_results_subdir}"
job_name = "{job_name}"
affinity_path = os.path.join(results_dir, "predictions", job_name, f"affinity_{{job_name}}.json")
if os.path.isfile(affinity_path):
    with open(affinity_path) as f:
        data = json.load(f)
    print(json.dumps(data))
else:
    print(json.dumps({{"error": "No affinity file found", "searched": affinity_path}}))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(affinity_script)
            tmp_script = f.name

        container_script = f"{self.workspace_dir}/.tmp_affinity_{call_number}.py"
        self._copy_to_container(tmp_script, container_script)
        os.unlink(tmp_script)

        affinity_output, _ = self._docker_exec(
            ["python", container_script],
            timeout=30,
        )
        self._docker_exec(["rm", "-f", container_script], timeout=10)

        # Parse affinity results
        try:
            affinity_data = json.loads(affinity_output.strip())
        except (json.JSONDecodeError, ValueError):
            affinity_data = {"raw_output": affinity_output.strip()[:500]}

        result = {
            "call_number": call_number,
            "budget_remaining": self.config.boltz_budget - self.boltz_calls,
            "cif_dir": f"boltz_results/call_{call_number}/boltz_results_{job_name}/predictions/{job_name}",
            "affinity_data": affinity_data,
        }

        # Extract key values for easy reading
        if isinstance(affinity_data, dict) and "error" not in affinity_data:
            bp = affinity_data.get("affinity_probability_binary")
            av = affinity_data.get("affinity_pred_value")
            if bp is not None:
                result["binding_probability"] = bp
            if av is not None:
                result["affinity_pred_value"] = av

        return json.dumps(result, indent=4), {
            "call_number": call_number,
            "budget_remaining": self.config.boltz_budget - self.boltz_calls,
        }

    def run_predict_admet(self, arguments):
        """Run ADMET-AI prediction inside the container."""
        if self.admet_calls >= self.config.admet_budget:
            return (
                f"ERROR: ADMET budget exceeded ({self.config.admet_budget} calls). "
                "No more ADMET predictions available.",
                {"budget_exceeded": True},
            )

        self.admet_calls += 1
        call_number = self.admet_calls

        smiles = arguments["smiles"]

        smiles_literal = json.dumps(smiles)
        admet_script = f"""
import json
import functools
import contextlib
import io
import torch
torch.load = functools.partial(torch.load, weights_only=False)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from admet_ai import ADMETModel
    model = ADMETModel()
    preds = model.predict(smiles={smiles_literal})
import sys
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
result = {{}}
column_map = {{
    "hERG": "herg",
    "BBB_Martins": "bbb",
    "Solubility_AqSolDB": "solubility",
    "CYP3A4_Veith": "cyp3a4",
    "Clearance_Hepatocyte_AZ": "clearance",
    "Caco2_Wang": "caco2",
    "PPBR_AZ": "ppb",
    "AMES": "ames",
}}
for col, name in column_map.items():
    for pred_col in preds:
        if col.lower() in pred_col.lower():
            val = preds[pred_col]
            result[name] = float(val.iloc[0] if hasattr(val, 'iloc') else val)
            break
print(json.dumps(result))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(admet_script)
            tmp_path = f.name

        container_script = f"{self.workspace_dir}/.tmp_admet_{call_number}.py"
        self._copy_to_container(tmp_path, container_script)
        os.unlink(tmp_path)

        try:
            output, returncode = self._docker_exec(
                ["/opt/oracle/bin/python", container_script],
                timeout=self.config.admet_timeout,
            )
            self._docker_exec_shell(f"rm -f {container_script}", timeout=10)
        except subprocess.TimeoutExpired:
            return (
                f"ERROR: ADMET prediction timed out after {self.config.admet_timeout} seconds.",
                {"call_number": call_number, "budget_remaining": self.config.admet_budget - self.admet_calls},
            )

        # Try to parse JSON from output
        try:
            result = json.loads(output.strip().split("\n")[-1])
            return json.dumps(result, indent=4), {
                "call_number": call_number,
                "budget_remaining": self.config.admet_budget - self.admet_calls,
            }
        except (json.JSONDecodeError, ValueError, IndexError):
            return self._truncate(output), {
                "call_number": call_number,
                "budget_remaining": self.config.admet_budget - self.admet_calls,
            }

    def run_dock_vina(self, arguments):
        """Run isolated Vina-GPU 2.1 with Meeko preparation."""
        if self.vina_calls >= self.config.vina_budget:
            return (
                f"ERROR: Vina budget exceeded ({self.config.vina_budget} calls).",
                {"budget_exceeded": True},
            )

        receptor_path, _ = self._resolve_workspace_path(
            arguments.get("receptor_path"), (".pdb", ".pdbqt")
        )
        center = arguments.get("center")
        box_size = arguments.get("box_size", [20.0, 20.0, 20.0])
        if not isinstance(center, list) or len(center) != 3:
            raise ValueError("center must be [x, y, z]")
        if not isinstance(box_size, list) or len(box_size) != 3:
            raise ValueError("box_size must be [x, y, z]")

        self.vina_calls += 1
        call_number = self.vina_calls
        output_dir = f"{self.workspace_dir}/vina_results/call_{call_number}"
        command = [
            "/opt/vina/bin/python", "/opt/smdd-tools/vina_dock.py",
            "--receptor", receptor_path,
            "--smiles", arguments["smiles"],
            "--output-dir", output_dir,
            "--center", *[str(value) for value in center],
            "--box-size", *[str(value) for value in box_size],
            "--search-depth", str(arguments.get("search_depth", 8)),
            "--num-poses", str(arguments.get("num_poses", 9)),
            "--seed", str(self.config.seed),
        ]
        try:
            output, returncode = self._exec_vina(
                command, timeout=self.config.vina_timeout
            )
        except subprocess.TimeoutExpired:
            return (
                f"ERROR: Vina docking timed out after {self.config.vina_timeout} seconds.",
                {"call_number": call_number,
                 "budget_remaining": self.config.vina_budget - self.vina_calls},
            )
        metadata = {
            "call_number": call_number,
            "budget_remaining": self.config.vina_budget - self.vina_calls,
        }
        if returncode != 0:
            return f"ERROR: Vina exited with code {returncode}.\n{self._truncate(output)}", metadata
        result = self._read_json_from_container(f"{output_dir}/result.json")
        result = self._make_workspace_paths_relative(result)
        result["budget_remaining"] = metadata["budget_remaining"]
        return json.dumps(result, indent=4), metadata

    def run_dock_diffdock(self, arguments):
        """Run isolated DiffDock-L v1.1.3 inference."""
        if self.diffdock_calls >= self.config.diffdock_budget:
            return (
                f"ERROR: DiffDock budget exceeded ({self.config.diffdock_budget} calls).",
                {"budget_exceeded": True},
            )

        receptor_path, _ = self._resolve_workspace_path(
            arguments.get("receptor_path"), (".pdb",)
        )
        self.diffdock_calls += 1
        call_number = self.diffdock_calls
        output_dir = f"{self.workspace_dir}/diffdock_results/call_{call_number}"
        command = [
            "/opt/diffdock/bin/python", "/opt/smdd-tools/diffdock_dock.py",
            "--receptor", receptor_path,
            "--smiles", arguments["smiles"],
            "--output-dir", output_dir,
            "--samples", str(arguments.get("num_samples", 10)),
            "--inference-steps", str(arguments.get("inference_steps", 20)),
        ]
        try:
            output, returncode = self._exec_diffdock(
                command, timeout=self.config.diffdock_timeout
            )
        except subprocess.TimeoutExpired:
            return (
                f"ERROR: DiffDock timed out after {self.config.diffdock_timeout} seconds.",
                {"call_number": call_number,
                 "budget_remaining": self.config.diffdock_budget - self.diffdock_calls},
            )
        metadata = {
            "call_number": call_number,
            "budget_remaining": self.config.diffdock_budget - self.diffdock_calls,
        }
        if returncode != 0:
            return f"ERROR: DiffDock exited with code {returncode}.\n{self._truncate(output)}", metadata
        result = self._read_json_from_container(f"{output_dir}/result.json")
        result = self._make_workspace_paths_relative(result)
        result["budget_remaining"] = metadata["budget_remaining"]
        return json.dumps(result, indent=4), metadata

    def run_submit(self, arguments):
        """Validate and register the agent's submission."""
        file_path = str(arguments.get("file_path", "")).strip()
        if file_path.startswith("/workspace/"):
            file_path = file_path[len("/workspace/"):]
        if not file_path or file_path.startswith("/"):
            return "ERROR: Submission path must be relative to the workspace.", {}
        file_path = posixpath.normpath(file_path)
        if file_path == ".." or file_path.startswith("../"):
            return "ERROR: Submission path must stay inside the workspace.", {}

        container_path = f"{self.workspace_dir}/{file_path}"

        _, returncode = self._docker_exec(["test", "-f", container_path], timeout=10)
        if returncode != 0:
            return f"ERROR: File not found at {container_path}. Please create the file first, then call submit again.", {}

        return f"Submission accepted: {file_path}", {"submitted_file": file_path}
