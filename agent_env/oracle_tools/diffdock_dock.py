"""Trusted wrapper around the pinned DiffDock-L inference CLI."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


CONFIDENCE_PATTERN = re.compile(r"_confidence(-?[0-9]+(?:\.[0-9]+)?)\.sdf$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--inference-steps", type=int, default=20)
    args = parser.parse_args()

    if not 1 <= args.samples <= 40:
        raise ValueError("samples must be in [1, 40]")
    if not 1 <= args.inference_steps <= 40:
        raise ValueError("inference_steps must be in [1, 40]")

    receptor = Path(args.receptor).resolve(strict=True)
    if receptor.suffix.lower() != ".pdb":
        raise ValueError("DiffDock receptor_path must point to a .pdb file")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "prediction"
    if prediction_dir.exists():
        shutil.rmtree(prediction_dir)
    for name in ("diffdock.log", "inference_config.yaml", "result.json"):
        (output_dir / name).unlink(missing_ok=True)

    app_dir = Path("/opt/diffdock/app")
    # DiffDock loads its YAML after parsing the CLI and therefore YAML values
    # override command-line values. Write a per-call config so the requested
    # sample count and inference steps are actually honored.
    with (app_dir / "default_inference_args.yaml").open(encoding="utf-8") as handle:
        inference_config = yaml.safe_load(handle)
    inference_config["samples_per_complex"] = args.samples
    inference_config["inference_steps"] = args.inference_steps
    # The released v1.1 config uses a 20-step schedule with 19 model updates.
    inference_config["actual_steps"] = max(1, args.inference_steps - 1)
    config_path = output_dir / "inference_config.yaml"
    config_path.write_text(
        yaml.safe_dump(inference_config, sort_keys=False),
        encoding="utf-8",
    )
    command = [
        "/opt/diffdock/bin/python",
        "-m", "inference",
        "--config", str(config_path),
        "--protein_path", str(receptor),
        "--ligand_description", args.smiles,
        "--complex_name", "prediction",
        "--out_dir", str(output_dir),
    ]
    environment = os.environ.copy()
    environment.update({
        "TORCH_HOME": "/opt/diffdock/cache/torch",
        "XDG_CACHE_HOME": "/opt/diffdock/cache/xdg",
        "PYTHONNOUSERSITE": "1",
    })
    completed = subprocess.run(
        command,
        cwd=app_dir,
        env=environment,
        capture_output=True,
        text=True,
    )
    log_path = output_dir / "diffdock.log"
    log_path.write_text(
        completed.stdout + ("\n--- stderr ---\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"DiffDock exited with {completed.returncode}: {detail[-4000:]}")

    pose_paths = sorted(prediction_dir.glob("rank*.sdf"))
    ranked_poses = []
    seen_ranks = set()
    for pose_path in pose_paths:
        rank_match = re.match(r"rank([0-9]+)", pose_path.name)
        if not rank_match:
            continue
        rank = int(rank_match.group(1))
        confidence_match = CONFIDENCE_PATTERN.search(pose_path.name)
        # Prefer confidence-bearing files over the duplicate plain rank1.sdf.
        if rank in seen_ranks and confidence_match is None:
            continue
        entry = {
            "rank": rank,
            "confidence": float(confidence_match.group(1)) if confidence_match else None,
            "path": str(pose_path),
        }
        if rank in seen_ranks:
            ranked_poses = [item for item in ranked_poses if item["rank"] != rank]
        ranked_poses.append(entry)
        seen_ranks.add(rank)
    ranked_poses.sort(key=lambda item: item["rank"])
    if not ranked_poses:
        raise RuntimeError("DiffDock completed without producing ranked SDF poses")
    if len(ranked_poses) != args.samples:
        raise RuntimeError(
            f"DiffDock produced {len(ranked_poses)} poses, expected {args.samples}"
        )

    result = {
        "tool": "diffdock",
        "version": "1.1.3",
        "model": "DiffDock-L v1.1",
        "samples": args.samples,
        "inference_steps": args.inference_steps,
        "actual_steps": inference_config["actual_steps"],
        "best_confidence": ranked_poses[0]["confidence"],
        "poses": ranked_poses,
        "log": str(log_path),
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DiffDock tool failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
