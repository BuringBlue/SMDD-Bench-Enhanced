"""Pharmacophore metrics with submitted code executed in a nested sandbox."""

import json
import os
import shutil
import sys
import tempfile

from rdkit import Chem

from smdd_runtime.isolation import Bind, Sandbox


RUNNER_CODE = r"""
import contextlib
import importlib.util
import json
import os
import sys

solution_path, inputs_path = sys.argv[1:]
with open(inputs_path) as handle:
    smiles = json.load(handle)
results = [None] * len(smiles)
try:
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        spec = importlib.util.spec_from_file_location("agent_solution", solution_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        check = module.check_pharmacophore
        for index, value in enumerate(smiles):
            try:
                result = check(value)
                results[index] = result if type(result) is bool else None
            except BaseException:
                pass
except BaseException:
    pass
sys.stdout.write(json.dumps(results))
"""


def _run_solution(agent_file_path, smiles):
    if not os.path.isfile(agent_file_path):
        raise FileNotFoundError(f"Agent solution file not found: {agent_file_path}")

    runtime = tempfile.mkdtemp(prefix="smdd_pharmacophore_")
    try:
        inputs_path = os.path.join(runtime, "inputs.json")
        with open(inputs_path, "w", encoding="utf-8") as handle:
            json.dump(smiles, handle)
        sandbox = Sandbox(
            os.path.join(runtime, "sandbox"),
            [Bind(agent_file_path), Bind(runtime, writable=True)],
            runtime,
            {"CUDA_VISIBLE_DEVICES": "", "HOME": runtime, "TMPDIR": runtime},
            new_session=False,
        )
        result = sandbox.run([sys.executable, "-c", RUNNER_CODE, agent_file_path, inputs_path], 120)
        if result.returncode:
            return [None] * len(smiles)
        values = json.loads(result.stdout)
        return values if len(values) == len(smiles) else [None] * len(smiles)
    except Exception:
        return [None] * len(smiles)
    finally:
        shutil.rmtree(runtime, ignore_errors=True)


def _read_smiles(path):
    with open(path, "r", encoding="utf-8") as handle:
        values = [line.strip().split()[0] for line in handle if line.strip()]
    return [value for value in values if Chem.MolFromSmiles(value) is not None]


def assess_recall(agent_file_path, hidden_actives_path):
    if not os.path.isfile(hidden_actives_path):
        raise FileNotFoundError(f"Hidden actives file not found: {hidden_actives_path}")
    smiles = _read_smiles(hidden_actives_path)
    if not smiles:
        return 0.0
    results = _run_solution(agent_file_path, smiles)
    return sum(result is True for result in results) / len(smiles)


def assess_specificity(agent_file_path, hidden_decoys_path):
    if not os.path.isfile(hidden_decoys_path):
        raise FileNotFoundError(f"Hidden decoys file not found: {hidden_decoys_path}")
    smiles = _read_smiles(hidden_decoys_path)
    if not smiles:
        return 0.0
    results = _run_solution(agent_file_path, smiles)
    return sum(result is False for result in results) / len(smiles)
