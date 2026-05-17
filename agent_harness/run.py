"""CLI entrypoint for the SMDD-Bench agent harness (single host, multiprocessing).

Runs tasks in parallel on one machine, one GPU per worker process.  Uses
``multiprocessing.Pool`` with the ``spawn`` start method so each worker gets
its own LiteLLM client / CUDA context / interpreter state.

The sandbox setup is identical to the internal Ray-cluster runner
(``run_mail.py``):

  * Tasks are obfuscated before being mounted into the container
    (target identifiers / PDB IDs stripped — see ``obfuscate.py``).
  * Containers run on a ``--internal`` Docker network with the MSA
    server reachable only through the nginx TCP proxy sidecar
    (see ``network.py``).
  * Each container is pinned to one GPU via ``--gpus device=N``.

Usage:
    python -m agent_harness.run --run-name my_run --gpus 0,1,2,3
    python -m agent_harness.run --run-name my_run --task-file my_tasks.txt
"""

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv

load_dotenv()

from agent_harness.agent_loop import run_agent
from agent_harness.config import AgentConfig
from agent_harness.network import COLABFOLD_HOST, NETWORK_NAME, ensure_network, start_proxy
from agent_harness.obfuscate import obfuscate_task, cleanup_obfuscated

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smdd_agent_harness")


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpus():
    """Return a list of physical GPU indices visible to nvidia-smi."""
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return []
    return [int(line.strip()) for line in proc.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Container lifecycle helpers
# ---------------------------------------------------------------------------

def start_container(docker_image, proxy_ip, gpu_id, task_dir=None, task_id=None):
    """Create and start a Docker container, returning the container ID.

    The container joins the internal network with DNS pointing
    api.colabfold.com at the proxy.  Input files are copied in via
    ``docker cp`` (no bind mount) so the agent gets a writable /workspace
    isolated from the host.
    """
    subprocess.run(["docker", "rm", "-f", task_id], capture_output=True)

    cmd = ["docker", "create", "-it", "--shm-size=32g"]
    if task_id:
        cmd += ["--name", task_id]
    cmd += ["--gpus", f'device={gpu_id}']
    cmd += ["--network", NETWORK_NAME]
    cmd += ["--add-host", f"{COLABFOLD_HOST}:{proxy_ip}"]
    cmd += ["--dns", "0.0.0.0"]
    cmd += [docker_image, "/bin/bash"]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to create container: {proc.stderr}")
    container_id = proc.stdout.strip()

    start_proc = subprocess.run(["docker", "start", container_id], capture_output=True)
    if start_proc.returncode != 0:
        raise RuntimeError(f"Failed to start container: {start_proc.stderr.decode()}")

    # Copy input files into /workspace.  task.yaml is excluded since it contains
    # evaluation config the agent must not see.
    for filename in os.listdir(task_dir):
        src = os.path.join(task_dir, filename)
        if os.path.isfile(src) and filename != "task.yaml":
            subprocess.run(
                ["docker", "cp", src, f"{container_id}:/workspace/{filename}"],
                check=True, capture_output=True,
            )

    return container_id


def stop_container(container_id):
    subprocess.run(["docker", "stop", container_id], capture_output=True)
    subprocess.run(["docker", "rm", container_id], capture_output=True)


def copy_output_from_container(container_id, output_file, local_dir):
    container_path = f"/workspace/{output_file}"
    local_path = os.path.join(local_dir, output_file)
    os.makedirs(os.path.dirname(local_path) or local_dir, exist_ok=True)
    subprocess.run(
        ["docker", "cp", f"{container_id}:{container_path}", local_path],
        check=True, capture_output=True,
    )
    return local_path


# ---------------------------------------------------------------------------
# Task resolution
# ---------------------------------------------------------------------------

def resolve_tasks(args):
    tasks_dir = args.tasks_dir

    if args.tasks:
        for tid in args.tasks:
            task_path = os.path.join(tasks_dir, tid, "task.yaml")
            if not os.path.isfile(task_path):
                print(f"Task not found: {task_path}", file=sys.stderr)
                sys.exit(1)
        return args.tasks

    if args.task_file:
        task_ids = []
        with open(args.task_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                task_ids.append(line)
        for tid in task_ids:
            task_path = os.path.join(tasks_dir, tid, "task.yaml")
            if not os.path.isfile(task_path):
                print(f"Task not found: {task_path}", file=sys.stderr)
                sys.exit(1)
        return task_ids

    task_ids = []
    for name in sorted(os.listdir(tasks_dir)):
        task_path = os.path.join(tasks_dir, name, "task.yaml")
        if os.path.isfile(task_path):
            task_ids.append(name)
    if not task_ids:
        print(f"No tasks found in {tasks_dir}", file=sys.stderr)
        sys.exit(1)
    return task_ids


# ---------------------------------------------------------------------------
# Single-task execution
# ---------------------------------------------------------------------------

def run_single_task(task_id, task_index, tasks_dir, config, run_results_dir, proxy_ip, gpu_id):
    """Run a single task end-to-end: obfuscate, container, agent, collect results."""
    task_dir = os.path.join(tasks_dir, task_id)
    results_dir = os.path.join(run_results_dir, task_id)
    os.makedirs(results_dir, exist_ok=True)

    obfuscated_dir = None
    container_id = None

    try:
        start_time = time.time()

        task_alias = f"task_{task_index:03d}"
        obfuscated_dir = obfuscate_task(task_dir, task_alias)
        mount_dir = obfuscated_dir

        container_id = start_container(
            config.docker_image, gpu_id=gpu_id, task_dir=mount_dir,
            proxy_ip=proxy_ip, task_id=task_id,
        )

        yaml_path = os.path.join(task_dir, "task.yaml")
        with open(yaml_path, "r") as f:
            task_yaml = yaml.safe_load(f)

        result = run_agent(task_yaml, mount_dir, config, container_id, results_dir)

        # Copy output file back from the container (if any).  Agent may have
        # created the file but never submitted it -- still try to grab it.
        output_file = task_yaml.get("output_config", {}).get("file_path")
        if result.submitted_file:
            output_file = result.submitted_file
        if output_file:
            try:
                copy_output_from_container(container_id, output_file, results_dir)
            except Exception:
                pass

        result_meta = {
            "task_id": task_id,
            "model": config.model,
            "status": result.status,
            "turns_used": result.turns_used,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "total_cache_read_tokens": result.total_cache_read_tokens,
            "total_cache_creation_tokens": result.total_cache_creation_tokens,
            "boltz_calls": result.boltz_calls,
            "admet_calls": result.admet_calls,
            "submitted_file": result.submitted_file,
            "error": result.error,
            "runtime_seconds": round(time.time() - start_time, 2),
        }
        with open(os.path.join(results_dir, "agent_result.json"), "w") as f:
            json.dump(result_meta, f, indent=4)

        return result_meta

    finally:
        if container_id:
            stop_container(container_id)
        if obfuscated_dir:
            cleanup_obfuscated(obfuscated_dir)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(run_name, run_results_dir, model):
    """Scan all agent_result.json files and assemble agent_summary.json."""
    tasks = {}
    by_type = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read_tokens = 0
    total_cache_creation_tokens = 0
    total_boltz = 0
    total_admet = 0

    for task_dir_name in sorted(os.listdir(run_results_dir)):
        result_path = os.path.join(run_results_dir, task_dir_name, "agent_result.json")
        if not os.path.isfile(result_path):
            continue

        with open(result_path, "r") as f:
            result = json.load(f)

        task_id = result.get("task_id", task_dir_name)
        status = result.get("status", "error")
        input_tok = result.get("total_input_tokens", 0)
        output_tok = result.get("total_output_tokens", 0)
        cache_read_tok = result.get("total_cache_read_tokens", 0)
        cache_creation_tok = result.get("total_cache_creation_tokens", 0)
        tasks[task_id] = {
            "status": status,
            "turns": result.get("turns_used", 0),
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "tokens": input_tok + output_tok,
            "cache_read_tokens": cache_read_tok,
            "cache_creation_tokens": cache_creation_tok,
            "boltz_calls": result.get("boltz_calls", 0),
            "admet_calls": result.get("admet_calls", 0),
        }

        total_input_tokens += input_tok
        total_output_tokens += output_tok
        total_cache_read_tokens += cache_read_tok
        total_cache_creation_tokens += cache_creation_tok
        total_boltz += tasks[task_id]["boltz_calls"]
        total_admet += tasks[task_id]["admet_calls"]

        parts = task_id.split("_")
        if len(parts) >= 2:
            try:
                type_num = str(int(parts[1]))
            except ValueError:
                type_num = "unknown"
        else:
            type_num = "unknown"

        if type_num not in by_type:
            by_type[type_num] = {"submitted": 0, "max_turns_exceeded": 0, "token_budget_exceeded": 0, "error": 0, "total": 0}
        by_type[type_num]["total"] += 1
        if status in by_type[type_num]:
            by_type[type_num][status] += 1

    summary = {
        "run_name": run_name,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks_total": len(tasks),
        "tasks_submitted": sum(1 for t in tasks.values() if t["status"] == "submitted"),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cache_read_tokens": total_cache_read_tokens,
        "total_cache_creation_tokens": total_cache_creation_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_boltz_calls": total_boltz,
        "total_admet_calls": total_admet,
        "by_type": by_type,
        "tasks": tasks,
    }

    summary_path = os.path.join(run_results_dir, "agent_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    return summary


# ---------------------------------------------------------------------------
# Worker (runs in a multiprocessing child process)
# ---------------------------------------------------------------------------

def _worker(work_item):
    """Pool worker: pick a GPU off the shared queue, run one task, release the GPU.

    The GPU queue acts as a semaphore + assignment mechanism: with one slot
    per GPU the pool naturally caps concurrency at ``len(gpus)``.

    A failure in one task must not abort the run, so we catch at this boundary
    and return the exception.  The driver loop logs it and moves on (same
    behavior as the Ray runner's per-future try/except).
    """
    (task_id, task_index, tasks_dir, config_dict, run_results_dir,
     proxy_ip, gpu_queue) = work_item

    config = AgentConfig(**config_dict)
    gpu_id = gpu_queue.get()
    try:
        result = run_single_task(
            task_id, task_index, tasks_dir, config,
            run_results_dir, proxy_ip, gpu_id,
        )
        return (task_id, result)
    except Exception as e:
        return (task_id, e)
    finally:
        gpu_queue.put(gpu_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SMDD-Bench agent harness (single host, multiprocessing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-name", required=True,
                        help="Name for this run; results go to {results-dir}/{run-name}/")
    parser.add_argument("--tasks-dir", default="tasks",
                        help="Path to tasks directory")
    parser.add_argument("--results-dir", default="results",
                        help="Path for results output")

    parser.add_argument("--tasks", nargs="+",
                        help="Run specific task IDs (default: all tasks)")
    parser.add_argument("--task-file",
                        help="File with task IDs to run (one per line)")
    parser.add_argument("--model",
                        help="Model name to use (overrides AgentConfig default)")

    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices, e.g. '0,1,2,3' "
                             "(default: all GPUs visible to nvidia-smi)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes "
                             "(default: number of GPUs)")

    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)

    args.tasks_dir = os.path.abspath(args.tasks_dir)
    args.results_dir = os.path.abspath(args.results_dir)

    task_ids = resolve_tasks(args)
    logger.info(f"Tasks to run: {len(task_ids)}")

    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        gpu_ids = detect_gpus()
    if not gpu_ids:
        print("No GPUs detected.  Agent tasks require a GPU for Boltz; aborting.",
              file=sys.stderr)
        sys.exit(1)
    logger.info(f"GPUs available: {gpu_ids}")

    n_workers = args.workers if args.workers is not None else len(gpu_ids)

    config = AgentConfig()
    if args.model:
        config.model = args.model
    config_dict = asdict(config)

    run_results_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(run_results_dir, exist_ok=True)
    with open(os.path.join(run_results_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    # Host-side proxy + internal network setup.  Single machine, no node-local
    # lock needed (run_mail.py uses fcntl to serialize startup across Ray
    # workers landing on the same node -- not applicable here).
    ensure_network()
    proxy_ip = start_proxy()
    logger.info(f"MSA proxy on {NETWORK_NAME} at {proxy_ip}")

    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)

    work_items = [
        (task_id, idx, args.tasks_dir, config_dict, run_results_dir,
         proxy_ip, gpu_queue)
        for idx, task_id in enumerate(task_ids)
    ]

    n_done = 0
    n_error = 0
    t0 = time.monotonic()
    with multiprocessing.Pool(processes=n_workers) as pool:
        try:
            # Worker returns (task_id, result_or_exception) so imap_unordered's
            # out-of-order results can still be attributed.
            for task_id, result in pool.imap_unordered(_worker, work_items):
                n_done += 1
                elapsed = time.monotonic() - t0
                if isinstance(result, BaseException):
                    n_error += 1
                    logger.error(f"[{n_done}/{len(task_ids)}] ERROR {task_id} ({elapsed:.0f}s): {result}")
                else:
                    status = result.get("status", "unknown")
                    logger.info(f"[{n_done}/{len(task_ids)}] DONE {task_id} -> {status} ({elapsed:.0f}s)")
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise

    summary = build_summary(args.run_name, run_results_dir, config.model)

    print(f"\n{'=' * 60}")
    print(f"Run: {args.run_name}")
    print(f"Model: {config.model}")
    print(f"Tasks: {summary['tasks_total']}")
    print(f"Submitted: {summary['tasks_submitted']}/{summary['tasks_total']}")
    print(f"Worker errors: {n_error}")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Cache read tokens: {summary['total_cache_read_tokens']:,}")
    print(f"Cache creation tokens: {summary['total_cache_creation_tokens']:,}")
    print(f"Total Boltz calls: {summary['total_boltz_calls']}")
    print(f"Total ADMET calls: {summary['total_admet_calls']}")
    print(f"\nBy type:")
    for type_num, counts in sorted(summary["by_type"].items()):
        print(f"  Type {type_num}: {counts['submitted']}/{counts['total']} submitted")
    print(f"\nSummary: {run_results_dir}/agent_summary.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
