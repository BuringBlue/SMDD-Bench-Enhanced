"""Docking metrics backed by the isolated Vina and DiffDock environments."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

from rdkit import Chem

from metrics.lead_optimization import read_smiles_file


VINA_PYTHON = "/opt/vina/bin/python"
VINA_ENTRYPOINT = "/opt/smdd-tools/vina_dock.py"
DIFFDOCK_PYTHON = "/opt/diffdock/bin/python"
DIFFDOCK_ENTRYPOINT = "/opt/smdd-tools/diffdock_dock.py"


def _run_json_command(command: list[str], timeout: int) -> dict:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"{Path(command[0]).name} failed: {detail[-4000:]}")

    for line in reversed(completed.stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    raise RuntimeError(f"{Path(command[0]).name} returned no JSON result")


def _docking_box_from_sdf(
    reference_file_path: str,
    padding: float = 4.0,
    minimum_size: float = 18.0,
) -> tuple[list[float], list[float]]:
    supplier = Chem.SDMolSupplier(reference_file_path, removeHs=False)
    reference = next((molecule for molecule in supplier if molecule is not None), None)
    if reference is None or reference.GetNumConformers() == 0:
        raise ValueError("Reference SDF does not contain a valid 3D molecule")

    positions = reference.GetConformer().GetPositions()
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    center = ((lower + upper) / 2).tolist()
    size = [max(minimum_size, float(span) + 2 * padding) for span in upper - lower]
    return [float(value) for value in center], size


def assess_vina_task4(
    agent_file_path: str,
    receptor_file_path: str,
    reference_file_path: str,
    output_dir: str,
    max_affinity_kcal_mol: float | None = None,
    search_depth: int = 8,
    num_poses: int = 9,
    timeout: int = 900,
) -> dict:
    """Dock a type-4 candidate with Vina and optionally apply an affinity cutoff."""
    center, box_size = _docking_box_from_sdf(reference_file_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = _run_json_command(
        [
            VINA_PYTHON,
            VINA_ENTRYPOINT,
            "--receptor",
            receptor_file_path,
            "--smiles",
            read_smiles_file(agent_file_path),
            "--output-dir",
            str(output_path),
            "--center",
            *(str(value) for value in center),
            "--box-size",
            *(str(value) for value in box_size),
            "--search-depth",
            str(search_depth),
            "--num-poses",
            str(num_poses),
        ],
        timeout=timeout,
    )

    raw_affinity = result.get("best_affinity_kcal_mol")
    if raw_affinity is None:
        raise RuntimeError("Vina returned no finite affinity")
    affinity = float(raw_affinity)
    if not math.isfinite(affinity):
        raise RuntimeError("Vina returned no finite affinity")
    passed = max_affinity_kcal_mol is None or affinity <= max_affinity_kcal_mol
    return {
        "all_passed": passed,
        "best_affinity_kcal_mol": affinity,
        "max_affinity_kcal_mol": max_affinity_kcal_mol,
        "center": result.get("center", center),
        "box_size": result.get("box_size", box_size),
        "backend": result.get("backend"),
        "gpu_device": result.get("gpu_device"),
        "search_depth": result.get("search_depth", search_depth),
        "energies": result.get("energies", []),
        "poses": {
            "pdbqt": "vina/poses.pdbqt",
            "sdf": "vina/poses.sdf" if result.get("poses_sdf") else None,
        },
    }


def assess_diffdock_task4(
    agent_file_path: str,
    receptor_file_path: str,
    output_dir: str,
    min_confidence: float | None = None,
    samples: int = 10,
    inference_steps: int = 20,
    timeout: int = 3600,
) -> dict:
    """Dock a type-4 candidate with DiffDock and optionally apply a confidence cutoff."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = _run_json_command(
        [
            DIFFDOCK_PYTHON,
            DIFFDOCK_ENTRYPOINT,
            "--receptor",
            receptor_file_path,
            "--smiles",
            read_smiles_file(agent_file_path),
            "--output-dir",
            str(output_path),
            "--samples",
            str(samples),
            "--inference-steps",
            str(inference_steps),
        ],
        timeout=timeout,
    )

    raw_confidence = result.get("best_confidence")
    if raw_confidence is None:
        raise RuntimeError("DiffDock returned no finite confidence")
    confidence = float(raw_confidence)
    if not math.isfinite(confidence):
        raise RuntimeError("DiffDock returned no finite confidence")
    passed = min_confidence is None or confidence >= min_confidence
    poses = [
        {
            "rank": pose.get("rank"),
            "confidence": pose.get("confidence"),
            "path": f"diffdock/prediction/{Path(pose['path']).name}",
        }
        for pose in result.get("poses", [])
    ]
    return {
        "all_passed": passed,
        "best_confidence": confidence,
        "min_confidence": min_confidence,
        "samples": result.get("samples", samples),
        "inference_steps": result.get("inference_steps", inference_steps),
        "actual_steps": result.get("actual_steps"),
        "poses": poses,
        "log": "diffdock/diffdock.log",
    }
