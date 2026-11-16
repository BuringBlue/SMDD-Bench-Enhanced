"""Trusted Vina-GPU 2.1 entrypoint used by the host-side tool executor."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


VINA_GPU_BINARY = Path("/opt/vina/bin/AutoDock-Vina-GPU-2-1")
VINA_GPU_ASSETS = Path("/opt/vina/share/vina-gpu")
VINA_RESULT_PATTERN = re.compile(
    r"^REMARK VINA RESULT:\s+(-?[0-9]+(?:\.[0-9]+)?)"
    r"\s+(-?[0-9]+(?:\.[0-9]+)?)\s+(-?[0-9]+(?:\.[0-9]+)?)",
    re.MULTILINE,
)


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Command failed ({command[0]}): {detail[-4000:]}")


def _bounded_vector(values: list[float], name: str, minimum: float, maximum: float) -> list[float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) and minimum <= value <= maximum for value in vector):
        raise ValueError(f"{name} values must be finite and in [{minimum}, {maximum}]")
    return vector


def _prepare_ligand(smiles: str, output_dir: Path, seed: int) -> Path:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("SMILES could not be parsed by RDKit")
    molecule = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(molecule, params) != 0:
        raise ValueError("RDKit failed to generate a 3D ligand conformer")
    try:
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(molecule, maxIters=500)

    sdf_path = output_dir / "ligand_input.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(molecule)
    writer.close()

    pdbqt_path = output_dir / "ligand.pdbqt"
    _run([
        "/opt/vina/bin/mk_prepare_ligand.py",
        "-i", str(sdf_path),
        "-o", str(pdbqt_path),
    ])
    return pdbqt_path


def _prepare_receptor(receptor_path: Path, output_dir: Path) -> Path:
    if receptor_path.suffix.lower() == ".pdbqt":
        return receptor_path
    if receptor_path.suffix.lower() != ".pdb":
        raise ValueError("Vina receptor_path must point to a .pdb or .pdbqt file")

    cleaned_path = output_dir / "receptor_protein_only.pdb"
    kept = 0
    with receptor_path.open("r", encoding="utf-8", errors="replace") as source, cleaned_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if line.startswith("ATOM  "):
                destination.write(line)
                kept += 1
            elif line.startswith("TER"):
                destination.write(line)
        destination.write("END\n")
    if kept == 0:
        raise ValueError("Receptor PDB contains no ATOM records")

    receptor_prefix = output_dir / "receptor"
    _run([
        "/opt/vina/bin/mk_prepare_receptor.py",
        "--read_pdb", str(cleaned_path),
        "-o", str(receptor_prefix),
        "-p",
    ])
    pdbqt_path = receptor_prefix.with_suffix(".pdbqt")
    if not pdbqt_path.is_file():
        raise RuntimeError("Meeko did not produce receptor.pdbqt")
    return pdbqt_path


def _parse_vina_results(poses_path: Path) -> list[list[float]]:
    text = poses_path.read_text(encoding="utf-8", errors="replace")
    energies = [[float(value) for value in match] for match in VINA_RESULT_PATTERN.findall(text)]
    if not energies or not all(math.isfinite(value) for row in energies for value in row):
        raise RuntimeError("Vina-GPU produced no finite docking affinities")
    return energies


def _run_vina_gpu(
    receptor: Path,
    ligand: Path,
    output_dir: Path,
    center: list[float],
    box_size: list[float],
    search_depth: int,
    num_poses: int,
    thread: int,
    seed: int,
) -> tuple[Path, str]:
    poses_path = output_dir / "poses.pdbqt"
    opencl_link = output_dir / "OpenCL"
    if opencl_link.exists() or opencl_link.is_symlink():
        raise RuntimeError(f"Temporary OpenCL path already exists: {opencl_link}")
    opencl_link.symlink_to(VINA_GPU_ASSETS / "OpenCL", target_is_directory=True)
    command = [
        str(VINA_GPU_BINARY),
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--out", str(poses_path),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--search_depth", str(search_depth),
        "--thread", str(thread),
        "--num_modes", str(num_poses),
        "--seed", str(seed),
        "--opencl_binary_path", str(output_dir),
    ]
    environment = os.environ.copy()
    environment.update({
        "OCL_ICD_VENDORS": str(VINA_GPU_ASSETS / "nvidia.icd"),
        "PYTHONNOUSERSITE": "1",
    })
    try:
        completed = subprocess.run(
            command,
            cwd=output_dir,
            env=environment,
            capture_output=True,
            text=True,
        )
    finally:
        opencl_link.unlink(missing_ok=True)
        for name in ("Kernel1_Opt.bin", "Kernel2_Opt.bin"):
            (output_dir / name).unlink(missing_ok=True)

    log = completed.stdout + ("\n--- stderr ---\n" + completed.stderr if completed.stderr else "")
    (output_dir / "vina-gpu.log").write_text(log, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Vina-GPU exited with {completed.returncode}: {log.strip()[-4000:]}")
    if not poses_path.is_file():
        raise RuntimeError("Vina-GPU completed without producing poses.pdbqt")
    return poses_path, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--center", type=float, nargs=3, required=True)
    parser.add_argument("--box-size", type=float, nargs=3, default=(20.0, 20.0, 20.0))
    parser.add_argument("--search-depth", type=int, default=8)
    parser.add_argument("--num-poses", type=int, default=9)
    parser.add_argument("--thread", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    center = _bounded_vector(args.center, "center", -10000.0, 10000.0)
    box_size = _bounded_vector(args.box_size, "box_size", 1.0, 48.0)
    if not 1 <= args.search_depth <= 128:
        raise ValueError("search_depth must be in [1, 128]")
    if not 1 <= args.num_poses <= 50:
        raise ValueError("num_poses must be in [1, 50]")
    if not 1000 <= args.thread <= 100000:
        raise ValueError("thread must be in [1000, 100000]")

    receptor_path = Path(args.receptor).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "ligand_input.sdf",
        "ligand.pdbqt",
        "poses.pdbqt",
        "poses.sdf",
        "receptor.pdbqt",
        "receptor_protein_only.pdb",
        "result.json",
        "vina-gpu.log",
        "gpu_runtime.log",
    ):
        (output_dir / name).unlink(missing_ok=True)

    ligand_pdbqt = _prepare_ligand(args.smiles, output_dir, args.seed)
    receptor_pdbqt = _prepare_receptor(receptor_path, output_dir)
    poses_path, log = _run_vina_gpu(
        receptor_pdbqt,
        ligand_pdbqt,
        output_dir,
        center,
        box_size,
        args.search_depth,
        args.num_poses,
        args.thread,
        args.seed,
    )
    energies = _parse_vina_results(poses_path)

    poses_sdf = output_dir / "poses.sdf"
    export_error = None
    try:
        _run([
            "/opt/vina/bin/mk_export.py",
            str(poses_path),
            "-s", str(poses_sdf),
        ])
    except Exception as exc:
        export_error = str(exc)

    platform_match = re.search(r"^GPU Platform:\s*(.+)$", log, re.MULTILINE)
    device_match = re.search(r"^GPU Device:\s*(.+)$", log, re.MULTILINE)
    result = {
        "tool": "vina",
        "backend": "vina-gpu-2.1",
        "version": "2.1",
        "opencl_platform": platform_match.group(1).strip() if platform_match else None,
        "gpu_device": device_match.group(1).strip() if device_match else None,
        "best_affinity_kcal_mol": energies[0][0],
        "energies": energies,
        "energy_columns": ["affinity_kcal_mol", "rmsd_lb", "rmsd_ub"],
        "center": center,
        "box_size": box_size,
        "search_depth": args.search_depth,
        "thread": args.thread,
        "num_poses": len(energies),
        "poses_pdbqt": str(poses_path),
        "poses_sdf": str(poses_sdf) if poses_sdf.is_file() else None,
        "log": str(output_dir / "vina-gpu.log"),
    }
    if export_error:
        result["sdf_export_warning"] = export_error

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Vina tool failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
