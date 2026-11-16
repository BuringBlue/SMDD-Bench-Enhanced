"""Host-side CLI orchestrator for SMDD-Bench evaluation (single host, multiprocessing).

Launches one Docker container per task (one task per GPU), then assembles a
``summary.json`` from all per-task ``result.json`` files.

Uses ``multiprocessing.Pool`` with the ``spawn`` start method.  A GPU
``Queue`` doubles as both an assignment mechanism and a semaphore: with one
slot per GPU the pool naturally caps concurrency.

If no GPUs are detected (CPU-only host), tasks are run sequentially.
"""

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from smdd_runtime.run_state import agent_run_is_complete

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smdd_evaluator")

TERMINAL_TASK_STATUSES = {"passed", "failed"}
AGENT_SUBMITTED_STATUS = "submitted"
AGENT_NOT_SUBMITTED_REASON = "agent_not_submitted"
AGENT_TERMINAL_FAILURE_STATUSES = {
    "max_turns_exceeded",
    "token_budget_exceeded",
    "no_tool_limit_exceeded",
    "error",
}


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpus():
    """Return list of GPU device indices via nvidia-smi, or [] if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
        )
        return [int(line.strip()) for line in out.strip().splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


# ---------------------------------------------------------------------------
# Task resolution
# ---------------------------------------------------------------------------

def resolve_tasks(args):
    """Return list of task IDs based on CLI selection flags."""
    tasks_dir = args.tasks_dir

    if args.all:
        task_ids = []
        for name in sorted(os.listdir(tasks_dir)):
            task_path = os.path.join(tasks_dir, name, "task.yaml")
            if os.path.isfile(task_path):
                task_ids.append(name)
        if not task_ids:
            print(f"No tasks found in {tasks_dir}", file=sys.stderr)
            sys.exit(1)
        return task_ids

    if args.all_in_folder:
        agent_outputs = os.path.abspath(args.agent_outputs)
        task_ids = []
        if not os.path.isdir(agent_outputs):
            return task_ids
        for name in sorted(os.listdir(agent_outputs)):
            if os.path.isdir(os.path.join(agent_outputs, name)):
                task_path = os.path.join(tasks_dir, name, "task.yaml")
                if os.path.isfile(task_path):
                    task_ids.append(name)
        return task_ids

    if args.tasks:
        for tid in args.tasks:
            task_path = os.path.join(tasks_dir, tid, "task.yaml")
            if not os.path.isfile(task_path):
                print(f"Task not found: {task_path}", file=sys.stderr)
                sys.exit(1)
        return args.tasks

    if args.task_file:
        task_ids = []
        with open(args.task_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                task_ids.append(line)
        for tid in task_ids:
            task_path = os.path.join(tasks_dir, tid, "task.yaml")
            if not os.path.isfile(task_path):
                print(f"Task not found: {task_path}", file=sys.stderr)
                sys.exit(1)
        return task_ids

    if args.type is not None:
        type_num = args.type
        prefix = f"smdd_{type_num:03d}"
        task_ids = []
        for name in sorted(os.listdir(tasks_dir)):
            if name == prefix or name.startswith(prefix + "_"):
                task_path = os.path.join(tasks_dir, name, "task.yaml")
                if os.path.isfile(task_path):
                    task_ids.append(name)
        if not task_ids:
            print(f"No tasks found matching type {type_num} (prefix: {prefix})", file=sys.stderr)
            sys.exit(1)
        return task_ids

    print("No task selection provided", file=sys.stderr)
    sys.exit(1)


def _agent_status(agent_outputs, task_id):
    """Return the validated agent status used to gate metric execution."""
    result_path = os.path.join(agent_outputs, task_id, "agent_result.json")
    try:
        with open(result_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
    except FileNotFoundError:
        return "missing_agent_result"
    except (OSError, ValueError, TypeError):
        return "invalid_agent_result"

    if not isinstance(result, dict) or result.get("task_id") != task_id:
        return "invalid_agent_result"
    status = result.get("status")
    return status if isinstance(status, str) and status else "invalid_agent_status"


def _write_not_submitted_result(run_results_dir, task_id, agent_status):
    """Record a zero-score result without running any evaluation metric."""
    result_dir = os.path.join(run_results_dir, task_id)
    os.makedirs(result_dir, exist_ok=True)
    result = {
        "task_id": task_id,
        "status": "failed",
        "reason_code": AGENT_NOT_SUBMITTED_REASON,
        "agent_status": agent_status,
        "metrics_executed": False,
        "steps": [
            {
                "name": "Agent submission",
                "status": "failed",
                "result": agent_status,
                "condition": {"eq": AGENT_SUBMITTED_STATUS},
                "duration_s": 0.0,
                "reason": (
                    f"Agent status is {agent_status!r}; metrics were not executed."
                ),
            }
        ],
    }
    result_path = os.path.join(result_dir, "result.json")
    temp_path = result_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=4)
    os.replace(temp_path, result_path)


def _remove_synthetic_not_submitted_result(run_results_dir, task_id):
    """Remove an obsolete synthetic failure while an Agent task is in flight."""
    result_path = os.path.join(run_results_dir, task_id, "result.json")
    try:
        with open(result_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, ValueError, TypeError):
        return
    if result.get("reason_code") == AGENT_NOT_SUBMITTED_REASON:
        try:
            os.unlink(result_path)
        except FileNotFoundError:
            pass


def filter_existing_agent_tasks(task_ids, agent_outputs):
    """Limit every selection mode to task folders already created by Agent."""
    return [
        task_id
        for task_id in task_ids
        if os.path.isdir(os.path.join(agent_outputs, task_id))
    ]


def prepare_evaluation_tasks(task_ids, agent_outputs, run_results_dir):
    """Partition submitted, terminal-failure, and still-running Agent tasks."""
    submitted = []
    not_submitted = {}
    incomplete = {}
    for task_id in task_ids:
        status = _agent_status(agent_outputs, task_id)
        if status == AGENT_SUBMITTED_STATUS:
            submitted.append(task_id)
        elif status in AGENT_TERMINAL_FAILURE_STATUSES:
            not_submitted[task_id] = status
            _write_not_submitted_result(
                run_results_dir, task_id, status
            )
        else:
            incomplete[task_id] = status
            _remove_synthetic_not_submitted_result(run_results_dir, task_id)
    return submitted, not_submitted, incomplete


def task_is_complete(run_results_dir, task_id):
    """Return true only for a valid, non-error terminal evaluation result."""
    result_path = os.path.join(run_results_dir, task_id, "result.json")
    try:
        with open(result_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(result, dict) or result.get("task_id") != task_id:
        return False
    # A later agent resume may change this task to submitted.  Its synthetic
    # zero-score result must never suppress the first real metric evaluation.
    if result.get("reason_code") == AGENT_NOT_SUBMITTED_REASON:
        return False
    if result.get("status") not in TERMINAL_TASK_STATUSES:
        return False
    steps = result.get("steps")
    if not isinstance(steps, list) or not steps or not all(
        isinstance(step, dict)
        and isinstance(step.get("name"), str)
        and step.get("status") in {"passed", "failed", "errored", "skipped"}
        for step in steps
    ):
        return False
    step_statuses = {step["status"] for step in steps}
    if "errored" in step_statuses:
        return False
    if result["status"] == "passed":
        return step_statuses == {"passed"}
    return "failed" in step_statuses


# ---------------------------------------------------------------------------
# Docker execution
# ---------------------------------------------------------------------------

def run_task_in_docker(task_id, tasks_dir, agent_outputs, run_results_dir, docker_image, gpu_id=None):
    """Run a single task evaluation in a Docker container. Returns the result dict."""
    task_host = os.path.abspath(os.path.join(tasks_dir, task_id))
    output_host = os.path.abspath(os.path.join(agent_outputs, task_id))
    result_host = os.path.abspath(os.path.join(run_results_dir, task_id))

    os.makedirs(result_host, exist_ok=True)

    cmd = ["docker", "run", "--rm", "--shm-size=32g"]
    if gpu_id is not None:
        cmd += ["--gpus", f'device={gpu_id}']
    drugbank_host = os.environ.get("SMDD_DRUGBANK_STRUCTURES")
    if drugbank_host:
        drugbank_host = os.path.abspath(drugbank_host)
        if not os.path.isfile(drugbank_host):
            raise FileNotFoundError(
                f"SMDD_DRUGBANK_STRUCTURES does not exist: {drugbank_host}"
            )
        drugbank_container = f"/eval/novelty/{os.path.basename(drugbank_host)}"
        cmd += [
            "-v", f"{drugbank_host}:{drugbank_container}:ro",
            "-e", f"SMDD_DRUGBANK_STRUCTURES={drugbank_container}",
        ]
    cmd += [
        "-v", f"{task_host}:/eval/task:ro",
        "-v", f"{output_host}:/eval/output:ro",
        "-v", f"{result_host}:/eval/results:rw",
        docker_image,
        "python", "/eval/eval_entrypoint.py",
    ]

    result_json_path = os.path.join(result_host, "result.json")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if os.path.isfile(result_json_path):
            with open(result_json_path, 'r') as f:
                return json.load(f)

        result = {
            "task_id": task_id,
            "status": "errored",
            "steps": [],
            "error": f"Container exited with code {proc.returncode} but produced no result.json",
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
        }

    except subprocess.TimeoutExpired:
        result = {
            "task_id": task_id,
            "status": "errored",
            "steps": [],
            "error": "Container timed out after 3600 seconds",
        }
    except Exception as e:
        result = {
            "task_id": task_id,
            "status": "errored",
            "steps": [],
            "error": str(e),
        }

    with open(result_json_path, 'w') as f:
        json.dump(result, f, indent=4)

    return result


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(run_name, results_dir, *, is_final=True, task_ids=None):
    """Scan all result.json files in results_dir/run_name and write summary.json."""
    tasks = {}
    not_submitted_tasks = set()
    step_failure_counts = {}
    by_type = {}
    included_task_ids = set(task_ids) if task_ids is not None else None

    run_results_dir = os.path.join(results_dir, run_name)
    if not os.path.isdir(run_results_dir):
        return None

    for task_dir_name in sorted(os.listdir(run_results_dir)):
        if included_task_ids is not None and task_dir_name not in included_task_ids:
            continue
        result_path = os.path.join(run_results_dir, task_dir_name, "result.json")
        if not os.path.isfile(result_path):
            continue

        with open(result_path, 'r') as f:
            result = json.load(f)

        task_id = result.get("task_id", task_dir_name)
        status = result.get("status", "errored")
        tasks[task_id] = status
        if result.get("reason_code") == AGENT_NOT_SUBMITTED_REASON:
            not_submitted_tasks.add(task_id)
        elif result.get("metrics_executed", True):
            for step in result.get("steps", []):
                if not isinstance(step, dict) or step.get("status") != "failed":
                    continue
                step_name = step.get("name")
                if isinstance(step_name, str):
                    step_failure_counts[step_name] = (
                        step_failure_counts.get(step_name, 0) + 1
                    )

        parts = task_id.split("_")
        if len(parts) >= 2:
            try:
                type_num = str(int(parts[1]))
            except ValueError:
                type_num = "unknown"
        else:
            type_num = "unknown"

        if type_num not in by_type:
            by_type[type_num] = {
                "passed": 0,
                "failed": 0,
                "errored": 0,
                "not_submitted": 0,
                "total": 0,
            }
        by_type[type_num]["total"] += 1
        if status in ("passed", "failed", "errored"):
            by_type[type_num][status] += 1
        if task_id in not_submitted_tasks:
            by_type[type_num]["not_submitted"] += 1

    summary = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks_evaluated": len(tasks),
        "tasks_passed": sum(1 for s in tasks.values() if s == "passed"),
        "tasks_failed": sum(1 for s in tasks.values() if s == "failed"),
        "tasks_errored": sum(1 for s in tasks.values() if s == "errored"),
        "tasks_not_submitted": len(not_submitted_tasks),
        "tasks_metrics_executed": len(tasks) - len(not_submitted_tasks),
        "step_failure_counts": step_failure_counts,
        "status": "complete" if is_final else "in_progress",
        "is_final": bool(is_final),
        "final_score": (
            (sum(1 for s in tasks.values() if s == "passed") / len(tasks))
            if is_final and tasks else (0.0 if is_final else None)
        ),
        "by_type": by_type,
        "tasks": tasks,
    }

    with open(os.path.join(run_results_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)

    return summary


# ---------------------------------------------------------------------------
# Worker (runs in a multiprocessing child process)
# ---------------------------------------------------------------------------

def _worker(work_item):
    """Pool worker: pick a GPU, run the eval container, release the GPU.

    A failure must not abort the run, so we catch at this boundary and return
    the exception alongside the task_id.  The driver loop logs and continues.
    """
    (task_id, tasks_dir, agent_outputs, results_dir_run, docker_image,
     gpu_queue) = work_item

    gpu_id = gpu_queue.get()
    try:
        result = run_task_in_docker(
            task_id, tasks_dir, agent_outputs, results_dir_run,
            docker_image, gpu_id=gpu_id,
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
        description="SMDD-Bench evaluation orchestrator (single host, multiprocessing)",
    )
    parser.add_argument("--run-name", required=True, help="Name for this evaluation run")
    parser.add_argument("--agent-outputs", required=True,
                        help="Directory containing agent output directories")
    parser.add_argument("--tasks-dir", default="tasks",
                        help="Directory containing task directories (default: tasks)")
    parser.add_argument("--docker-image", default="smdd-evals",
                        help="Docker image name (default: smdd-evals)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices, e.g. '0,1,2,3' "
                             "(default: all GPUs visible to nvidia-smi; "
                             "CPU-only if none detected)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes "
                             "(default: number of GPUs)")

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Evaluate all tasks in --tasks-dir")
    selection.add_argument("--all-in-folder", action="store_true",
                           help="Evaluate all tasks found in --agent-outputs folder")
    selection.add_argument("--tasks", nargs="+", help="Evaluate specific task IDs")
    selection.add_argument("--task-file", help="File with task IDs (one per line)")
    selection.add_argument("--type", type=int,
                           help="Evaluate all tasks of a specific type number")

    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)

    selected_task_ids = filter_existing_agent_tasks(
        resolve_tasks(args),
        os.path.abspath(args.agent_outputs),
    )
    logger.info(f"Agent task folders selected: {len(selected_task_ids)}")

    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        gpu_ids = detect_gpus()

    results_base = "results"
    results_dir_run = os.path.join(results_base, args.run_name)
    os.makedirs(results_dir_run, exist_ok=True)
    task_ids, not_submitted, incomplete = prepare_evaluation_tasks(
        selected_task_ids,
        os.path.abspath(args.agent_outputs),
        results_dir_run,
    )
    logger.info(
        "Agent submitted=%d, terminal non-submitted=%d, incomplete=%d; only submitted tasks run metrics",
        len(task_ids),
        len(not_submitted),
        len(incomplete),
    )

    n_done = 0
    n_error = 0
    t0 = time.monotonic()

    if not gpu_ids:
        # CPU-only fallback: sequential.  Some eval steps (e.g. Boltz2) require a
        # GPU and will fail; this mode is mainly useful for tasks whose metrics
        # are pure RDKit / SMILES-string checks.
        logger.warning("No GPUs detected -- running sequentially (CPU-only).")
        for task_id in task_ids:
            t_task = time.monotonic()
            try:
                result = run_task_in_docker(
                    task_id, args.tasks_dir, args.agent_outputs,
                    results_dir_run, args.docker_image, gpu_id=None,
                )
                status = result.get("status", "unknown")
                logger.info(f"[{n_done+1}/{len(task_ids)}] DONE {task_id} -> {status} "
                            f"({time.monotonic() - t_task:.0f}s)")
            except Exception as e:
                n_error += 1
                logger.error(f"[{n_done+1}/{len(task_ids)}] ERROR {task_id} "
                             f"({time.monotonic() - t_task:.0f}s): {e}")
            n_done += 1
    else:
        logger.info(f"GPUs available: {gpu_ids}")
        n_workers = args.workers if args.workers is not None else len(gpu_ids)

        manager = multiprocessing.Manager()
        gpu_queue = manager.Queue()
        for gpu_id in gpu_ids:
            gpu_queue.put(gpu_id)

        work_items = [
            (task_id, args.tasks_dir, args.agent_outputs, results_dir_run,
             args.docker_image, gpu_queue)
            for task_id in task_ids
        ]

        with multiprocessing.Pool(processes=n_workers) as pool:
            try:
                for task_id, result in pool.imap_unordered(_worker, work_items):
                    n_done += 1
                    elapsed = time.monotonic() - t0
                    if isinstance(result, BaseException):
                        n_error += 1
                        logger.error(f"[{n_done}/{len(task_ids)}] ERROR {task_id} "
                                     f"({elapsed:.0f}s): {result}")
                    else:
                        status = result.get("status", "unknown")
                        logger.info(f"[{n_done}/{len(task_ids)}] DONE {task_id} -> "
                                    f"{status} ({elapsed:.0f}s)")
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise

    summary = build_summary(
        args.run_name,
        results_base,
        is_final=agent_run_is_complete(os.path.abspath(args.agent_outputs)),
        task_ids=selected_task_ids,
    )

    if summary:
        print(f"\n{'='*60}")
        print(f"Run: {args.run_name}")
        print(f"Passed: {summary['tasks_passed']}/{summary['tasks_evaluated']}")
        print(f"Failed: {summary['tasks_failed']}/{summary['tasks_evaluated']}")
        print(f"Errored: {summary['tasks_errored']}/{summary['tasks_evaluated']}")
        print(f"Not submitted: {summary['tasks_not_submitted']}/{summary['tasks_evaluated']}")
        print(f"Metrics executed: {summary['tasks_metrics_executed']}/{summary['tasks_evaluated']}")
        print(f"Final score: {summary['final_score'] if summary['is_final'] else 'pending (Agent still running)'}")
        print(f"Worker errors: {n_error}")
        print(f"\nBy type:")
        for type_num, counts in sorted(summary['by_type'].items()):
            print(f"  Type {type_num}: {counts['passed']}/{counts['total']} passed")
        print(f"\nSummary written to: results/{args.run_name}/summary.json")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
