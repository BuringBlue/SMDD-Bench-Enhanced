"""One supervised local agent task; launched only by run_local."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time

import yaml

from agent_harness.agent_loop import run_agent
from agent_harness.config import AgentConfig
from agent_harness.local_tools import LocalToolExecutor
from agent_harness.obfuscate import cleanup_obfuscated, obfuscate_task


def _write_json(path, value):
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=4)
    os.replace(temp_path, path)


def _safe_output(workspace, relative_path):
    if not relative_path or os.path.isabs(relative_path):
        return None
    candidate = os.path.realpath(os.path.join(workspace, os.path.normpath(relative_path)))
    if os.path.commonpath((os.path.realpath(workspace), candidate)) != os.path.realpath(workspace):
        return None
    return candidate if os.path.isfile(candidate) else None


def run_task(args):
    task_dir = os.path.join(args.tasks_dir, args.task_id)
    results_dir = os.path.join(args.run_results_dir, args.task_id)
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, "agent_result.json")
    obfuscated_dir = None
    runtime_dir = None
    started = time.time()

    try:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = AgentConfig(**json.load(handle))
        with open(os.path.join(task_dir, "task.yaml"), "r", encoding="utf-8") as handle:
            task_yaml = yaml.safe_load(handle)

        alias = f"task_{args.task_index:03d}"
        obfuscated_dir = obfuscate_task(task_dir, alias)
        runtime_dir = tempfile.mkdtemp(prefix=f"smdd_{alias}_")
        executor = LocalToolExecutor(
            obfuscated_dir,
            runtime_dir,
            args.msa_socket,
            config,
            boltz_socket=args.boltz_socket,
        )
        result = run_agent(task_yaml, obfuscated_dir, config, "local", results_dir, executor=executor)

        output_name = result.submitted_file or task_yaml.get("output_config", {}).get("file_path")
        output_path = _safe_output(obfuscated_dir, output_name)
        if output_path:
            destination = os.path.join(results_dir, os.path.normpath(output_name))
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(output_path, destination)

        metadata = {
            "task_id": args.task_id,
            "model": config.model,
            "backend": "local-sandbox",
            "status": result.status,
            "turns_used": result.turns_used,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "total_cache_read_tokens": result.total_cache_read_tokens,
            "total_cache_creation_tokens": result.total_cache_creation_tokens,
            "boltz_calls": result.boltz_calls,
            "admet_calls": result.admet_calls,
            "vina_calls": result.vina_calls,
            "diffdock_calls": result.diffdock_calls,
            "submitted_file": result.submitted_file,
            "error": result.error,
            "runtime_seconds": round(time.time() - started, 2),
        }
    except Exception as exc:
        metadata = {
            "task_id": args.task_id,
            "backend": "local-sandbox",
            "status": "error",
            "turns_used": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_creation_tokens": 0,
            "boltz_calls": 0,
            "admet_calls": 0,
            "vina_calls": 0,
            "diffdock_calls": 0,
            "submitted_file": None,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_seconds": round(time.time() - started, 2),
        }
    finally:
        if runtime_dir:
            shutil.rmtree(runtime_dir, ignore_errors=True)
        if obfuscated_dir:
            cleanup_obfuscated(obfuscated_dir)

    _write_json(result_path, metadata)
    return 0 if metadata["status"] != "error" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--run-results-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--msa-socket", required=True)
    # Optional so an already-running pre-persistent supervisor can continue
    # launching tasks with the legacy per-call Boltz process.
    parser.add_argument("--boltz-socket")
    raise SystemExit(run_task(parser.parse_args()))


if __name__ == "__main__":
    main()
