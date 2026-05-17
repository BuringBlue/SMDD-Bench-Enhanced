"""Tool definitions and execution for the SMDD-Bench agent.

Tools execute inside a Docker container via `docker exec`.
Budget tracking for predict_boltz and predict_admet is managed host-side.
"""

import json
import os
import subprocess
import tempfile
import time
import uuid

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

    def _docker_exec(self, cmd, timeout=120):
        """Run a command inside the container. Returns (stdout+stderr, returncode)."""
        full_cmd = ["docker", "exec", "--workdir", self.workspace_dir, self.container_id] + cmd
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

    def _docker_exec_shell(self, shell_cmd, timeout=120):
        """Run a shell command inside the container. Returns (stdout+stderr, returncode)."""
        full_cmd = ["docker", "exec", "--workdir", self.workspace_dir, self.container_id, "bash", "-c", shell_cmd]
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

    def dispatch(self, tool_name, arguments):
        """Dispatch a tool call. Returns (result_str, extra_log_data)."""
        try:
            if tool_name == "python":
                return self.run_python(arguments)
            elif tool_name == "predict_boltz":
                return self.run_predict_boltz(arguments)
            elif tool_name == "predict_admet":
                return self.run_predict_admet(arguments)
            elif tool_name == "submit":
                return self.run_submit(arguments)
            else:
                return f"Unknown tool: {tool_name}", {}
        except Exception as e:
            # Roll back budget if the call was counted but never actually ran
            if tool_name == "predict_boltz" and self.boltz_calls > 0:
                self.boltz_calls -= 1
            elif tool_name == "predict_admet" and self.admet_calls > 0:
                self.admet_calls -= 1

            budget_msg = ""
            if tool_name == "predict_boltz":
                budget_msg = f" You have {self.config.boltz_budget - self.boltz_calls} Boltz calls remaining."
            elif tool_name == "predict_admet":
                budget_msg = f" You have {self.config.admet_budget - self.admet_calls} ADMET calls remaining."

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
            self._docker_exec_shell(f"rm -f {container_script_path}", timeout=10)
        except subprocess.TimeoutExpired:
            output = f"ERROR: Python execution timed out after {self.config.python_timeout} seconds."
            returncode = -1
        finally:
            os.unlink(tmp_path)

        return self._truncate(output), {}

    def run_predict_boltz(self, arguments):
        """Run Boltz structure prediction inside the container."""
        if self.boltz_calls >= self.config.boltz_budget:
            return (
                f"ERROR: Boltz budget exceeded ({self.config.boltz_budget} calls). "
                "No more Boltz predictions available.",
                {"budget_exceeded": True},
            )

        self.boltz_calls += 1
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
        boltz_cmd = (
            f"/opt/oracle/bin/boltz predict {container_yaml} "
            f"--out_dir {container_out_dir} "
            f"--use_msa_server --use_potentials "
            f"--diffusion_samples {num_samples} "
            f"--seed {self.config.seed} "
            f"--diffusion_samples_affinity {num_samples} "
            f"--max_parallel_samples 2"
        )

        try:
            output, returncode = self._docker_exec_shell(
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
        self._docker_exec_shell(f"rm -f {container_script}", timeout=10)

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
    preds = model.predict(smiles="{smiles}")
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

    def run_submit(self, arguments):
        """Validate and register the agent's submission."""
        file_path = arguments.get("file_path", "")
        # Normalize: remove leading slash or /workspace/ prefix
        if file_path.startswith("/workspace/"):
            file_path = file_path[len("/workspace/"):]
        elif file_path.startswith("/"):
            file_path = file_path[1:]

        container_path = f"{self.workspace_dir}/{file_path}"

        # Check file exists
        output, returncode = self._docker_exec_shell(
            f'test -f "{container_path}" && echo "EXISTS" || echo "NOT_FOUND"',
            timeout=10,
        )

        if "NOT_FOUND" in output:
            return f"ERROR: File not found at {container_path}. Please create the file first, then call submit again.", {}

        return f"Submission accepted: {file_path}", {}
