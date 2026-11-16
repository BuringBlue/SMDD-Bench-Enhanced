"""Daemonless evaluator runner for a persistent smdd-evals container."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import socket
import sys
import tempfile
import threading
import time
from contextlib import ExitStack

import yaml

from evaluator.evaluate import (
    build_summary,
    detect_gpus,
    filter_existing_agent_tasks,
    prepare_evaluation_tasks,
    resolve_tasks,
    task_is_complete,
)
from smdd_runtime.boltz import cache_dir, require_cache
from smdd_runtime.boltz_worker import BoltzService
from smdd_runtime.isolation import Bind, Sandbox, SandboxTimeout, verify_isolation
from smdd_runtime.msa import MsaService, msa_cache_dir
from smdd_runtime.novelty import NoveltyService
from smdd_runtime.preflight import validate_gpu_ids
from smdd_runtime.run_state import agent_run_is_complete


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smdd_evaluator_local")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _error_result(task_id, message):
    return {"task_id": task_id, "status": "errored", "steps": [], "error": message}


def _write_result(path, result):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=4)


def assign_worker_gpus(gpu_ids, workers):
    """Assign evaluator workers round-robin, allowing GPU sharing."""
    if not gpu_ids:
        return [None] * workers
    return [gpu_ids[index % len(gpu_ids)] for index in range(workers)]


def unique_gpu_ids(worker_gpus):
    """Return the physical GPUs that need one shared Boltz service each."""
    return list(dict.fromkeys(gpu_id for gpu_id in worker_gpus if gpu_id is not None))


def run_task(
    task_id,
    args,
    run_results_dir,
    runtime_root,
    msa_socket,
    novelty_socket,
    boltz_socket,
    gpu_id,
):
    task_dir = os.path.abspath(os.path.join(args.tasks_dir, task_id))
    output_dir = os.path.abspath(os.path.join(args.agent_outputs, task_id))
    result_dir = os.path.abspath(os.path.join(run_results_dir, task_id))
    task_runtime = tempfile.mkdtemp(prefix="eval_", dir=runtime_root)
    home = os.path.join(task_runtime, "home")
    temp = os.path.join(task_runtime, "tmp")
    meta = os.path.join(task_runtime, "sandbox")
    code_dir = os.path.join(task_runtime, "code")
    for path in (home, temp, meta, code_dir, result_dir):
        os.makedirs(path, exist_ok=True)
    for package in ("evaluator", "metrics", "smdd_runtime"):
        os.symlink(os.path.join(REPO_ROOT, package), os.path.join(code_dir, package))

    binds = [
        Bind(os.path.join(REPO_ROOT, "evaluator"), writable=False),
        Bind(os.path.join(REPO_ROOT, "metrics"), writable=False),
        Bind(os.path.join(REPO_ROOT, "smdd_runtime"), writable=False),
        Bind(code_dir, writable=False),
        Bind(task_dir, writable=False),
        Bind(output_dir, writable=False),
        Bind(result_dir, writable=True),
        Bind(home, writable=True),
        Bind(temp, writable=True),
    ]
    boltz_cache = cache_dir()
    if os.path.isdir(boltz_cache):
        binds.append(Bind(boltz_cache, writable=False))
        os.symlink(boltz_cache, os.path.join(home, ".boltz"))

    env = {
        "CUDA_VISIBLE_DEVICES": "" if gpu_id is None or task_id.startswith("smdd_001_") else str(gpu_id),
        "HOME": home,
        "SMDD_BOLTZ_MAX_PARALLEL_SAMPLES": "1",
        "SMDD_BOLTZ_NUM_WORKERS": "0",
        "SMDD_BOLTZ_TIMEOUT": str(args.task_timeout),
        "SMDD_CODE_DIR": code_dir,
        # Keep trusted runtime helpers importable for lazy metric imports
        # (for example, BoltzManager imports smdd_runtime only at step 4).
        # eval_entrypoint also prepends SMDD_CODE_DIR, but PYTHONPATH makes the
        # same guarantee for alternate Python entrypoints and child imports.
        "PYTHONPATH": code_dir,
        "SMDD_BOLTZ_EXECUTABLE": args.boltz_executable,
        "SMDD_ORACLE_PYTHON": args.oracle_python,
        "SMDD_OUTPUT_DIR": output_dir,
        "SMDD_RESULTS_DIR": result_dir,
        "SMDD_TASK_DIR": task_dir,
        "TMPDIR": temp,
    }
    if boltz_socket:
        # The random socket lives in the evaluator supervisor's mode-0700
        # runtime directory.  It is supplied only to this trusted metric
        # process; nested submitted-code sandboxes clear the environment and
        # deny socket creation.
        env["SMDD_BOLTZ_SOCKET"] = boltz_socket
    if novelty_socket:
        env["SMDD_NOVELTY_SOCKET"] = novelty_socket
    if args.debug:
        env["SMDD_BOLTZ_DEBUG_DIR"] = os.path.join(result_dir, "boltz_debug")
    broker = None
    try:
        broker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        broker.connect(msa_socket)
        env["SMDD_MSA_FD"] = str(broker.fileno())
        sandbox = Sandbox(
            meta,
            binds,
            temp,
            env,
            pass_fds=(broker.fileno(),),
            namespaces=False,
            landlock=False,
            # Type-1's trusted metric process needs SETUID/SETGID only to
            # create the nested user namespace for submitted Python.
            drop_caps=not task_id.startswith("smdd_001_"),
            allow_unix_sockets=True,
        )
        command = [sys.executable, os.path.join(REPO_ROOT, "evaluator", "eval_entrypoint.py")]
        execution = sandbox.run(command, args.task_timeout)
        result_path = os.path.join(result_dir, "result.json")
        if os.path.isfile(result_path):
            with open(result_path, "r", encoding="utf-8") as handle:
                result = json.load(handle)
        else:
            result = _error_result(task_id, f"evaluator exited with code {execution.returncode}: {execution.stderr[-2000:]}")
            _write_result(result_dir, result)
    except SandboxTimeout:
        result = _error_result(task_id, f"evaluation timed out after {args.task_timeout} seconds")
        _write_result(result_dir, result)
    except Exception as exc:
        result = _error_result(task_id, f"{type(exc).__name__}: {exc}")
        _write_result(result_dir, result)
    finally:
        if broker:
            broker.close()
        shutil.rmtree(task_runtime, ignore_errors=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="SMDD evaluator local sandbox backend")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--agent-outputs", required=True)
    parser.add_argument("--tasks-dir", default="tasks")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--gpus")
    parser.add_argument(
        "--workers",
        type=int,
        help="Concurrent evaluator tasks; may exceed the selected GPU count",
    )
    parser.add_argument("--task-timeout", type=int, default=3600)
    parser.add_argument("--boltz-cpu-threads", type=int, default=16)
    parser.add_argument("--boltz-worker-startup-timeout", type=int, default=3600)
    parser.add_argument(
        "--oracle-python",
        default=os.environ.get("SMDD_ORACLE_PYTHON", "/opt/oracle/bin/python"),
    )
    parser.add_argument(
        "--boltz-executable",
        default=os.environ.get("SMDD_BOLTZ_EXECUTABLE", "/opt/oracle/bin/boltz"),
    )
    parser.add_argument("--debug", action="store_true",
                        help="Persist Boltz stdout/stderr in each task result directory")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--all-in-folder", action="store_true")
    selection.add_argument("--tasks", nargs="+")
    selection.add_argument("--task-file")
    selection.add_argument("--type", type=int)
    args = parser.parse_args()

    args.tasks_dir = os.path.abspath(args.tasks_dir)
    args.agent_outputs = os.path.abspath(args.agent_outputs)
    args.results_dir = os.path.abspath(args.results_dir)
    args.oracle_python = os.path.abspath(args.oracle_python)
    args.boltz_executable = os.path.abspath(args.boltz_executable)
    task_ids = filter_existing_agent_tasks(
        resolve_tasks(args),
        args.agent_outputs,
    )
    agent_complete_at_start = agent_run_is_complete(args.agent_outputs)
    run_results_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(run_results_dir, exist_ok=True)
    submitted_task_ids, not_submitted, incomplete = prepare_evaluation_tasks(
        task_ids,
        args.agent_outputs,
        run_results_dir,
    )
    pending_task_ids = [
        task_id for task_id in submitted_task_ids
        if not task_is_complete(run_results_dir, task_id)
    ]
    already_complete = len(submitted_task_ids) - len(pending_task_ids)
    logger.info(
        "Agent folders=%d, incomplete=%d, not submitted=%d, already evaluated=%d, pending metrics=%d",
        len(task_ids),
        len(incomplete),
        len(not_submitted),
        already_complete,
        len(pending_task_ids),
    )
    if not pending_task_ids:
        summary = build_summary(
            args.run_name,
            args.results_dir,
            is_final=agent_run_is_complete(args.agent_outputs),
            task_ids=task_ids,
        )
        if summary["is_final"]:
            logger.info("Agent run is complete; final summary refreshed at %s",
                        os.path.join(run_results_dir, "summary.json"))
        else:
            logger.info("Agent run is still in progress; partial summary has no final score")
        return

    if not os.path.isfile(args.oracle_python):
        raise SystemExit(f"Oracle Python not found: {args.oracle_python}")
    if not os.path.isfile(args.boltz_executable):
        raise SystemExit(f"Boltz executable not found: {args.boltz_executable}")

    try:
        detected_gpu_ids = detect_gpus()
        gpu_ids = (
            [int(value.strip()) for value in args.gpus.split(",")]
            if args.gpus
            else detected_gpu_ids
        )
        if gpu_ids:
            validate_gpu_ids(gpu_ids, detected_gpu_ids)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Startup preflight failed: {exc}") from exc

    workers = args.workers if args.workers is not None else max(1, len(gpu_ids))
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    if min(
        args.task_timeout,
        args.boltz_cpu_threads,
        args.boltz_worker_startup_timeout,
    ) < 1:
        raise SystemExit("Startup preflight failed: timeouts and CPU threads must be positive")
    worker_devices = assign_worker_gpus(gpu_ids, workers)
    assigned_gpu_ids = unique_gpu_ids(worker_devices)
    needs_boltz = False
    needs_novelty = False
    for task_id in pending_task_ids:
        with open(os.path.join(args.tasks_dir, task_id, "task.yaml"), "r", encoding="utf-8") as handle:
            task = yaml.safe_load(handle)
        steps = task.get("evaluation", [])
        needs_boltz = needs_boltz or any(
            "boltz" in (step.get("module", "") + step.get("function", "")).lower()
            for step in steps
        )
        needs_novelty = needs_novelty or any(
            "novelty" in (step.get("module", "") + step.get("function", "")).lower()
            for step in steps
        )
        if needs_boltz and needs_novelty:
            break
    boltz_worker_gpus = assigned_gpu_ids if needs_boltz else []
    if needs_boltz and not boltz_worker_gpus:
        raise SystemExit(
            "Startup preflight failed: Boltz evaluation requires at least one selected GPU"
        )
    if boltz_worker_gpus and workers > len(boltz_worker_gpus):
        logger.info(
            "%d evaluator workers share %d per-GPU Boltz workers (%s); "
            "Boltz requests targeting the same GPU will be serialized",
            workers,
            len(boltz_worker_gpus),
            boltz_worker_gpus,
        )

    with open(os.path.join(run_results_dir, "runtime.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "backend": "local-sandbox",
            "network": "none; trusted services use fixed Unix-socket brokers",
            "msa_server": "https://api.colabfold.com",
            "msa_cache": msa_cache_dir(),
            "msa_cache_scope": "persistent-trusted-only",
            "boltz_max_parallel_samples": 1,
            "boltz_num_workers": 0,
            "boltz_mode": "lazy-persistent-gpu-resident-per-gpu",
            "boltz_oom_fallback": "cpu-backed-one-model-at-a-time-with-one-retry",
            "boltz_workers": len(boltz_worker_gpus),
            "boltz_worker_gpus": boltz_worker_gpus,
            "boltz_cpu_threads": args.boltz_cpu_threads,
            "boltz_fail_fast": True,
            "boltz_jit_cache": "private-per-gpu-worker",
            "resume_mode": "automatic",
            "selected_tasks": len(task_ids),
            "agent_submitted_at_start": len(submitted_task_ids),
            "agent_not_submitted_at_start": len(not_submitted),
            "agent_incomplete_at_start": len(incomplete),
            "agent_run_complete_at_start": agent_complete_at_start,
            "already_complete_at_start": already_complete,
            "pending_at_start": len(pending_task_ids),
            "gpus": gpu_ids,
            "workers": workers,
            "worker_gpus": worker_devices,
            "debug": args.debug,
            "unix_sockets": "trusted evaluator only; submitted code denied",
            "novelty_broker": needs_novelty,
        }, handle, indent=4)

    runtime_root = tempfile.mkdtemp(prefix="smdd_eval_run_")
    started = time.monotonic()
    counters = {"done": 0, "errored": 0}
    lock = threading.Lock()
    try:
        denied_path = os.path.join(args.tasks_dir, pending_task_ids[0], "task.yaml")
        verify_isolation(os.path.join(runtime_root, "untrusted"), denied_path, sys.executable)
        verify_isolation(
            os.path.join(runtime_root, "trusted"), denied_path, sys.executable,
            allow_unix_sockets=True,
        )
        if needs_boltz:
            try:
                boltz_cache = require_cache()
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
        else:
            boltz_cache = None
        broker_dir = os.path.join(runtime_root, "broker")
        service_dir = os.path.join(runtime_root, "evaluator_services")
        os.makedirs(service_dir, mode=0o700, exist_ok=True)
        os.chmod(service_dir, 0o700)
        with MsaService(
            os.path.join(broker_dir, "msa.sock"),
            args.oracle_python,
            cache_dir=msa_cache_dir(),
        ) as msa_socket, ExitStack() as services_stack:
            novelty_socket = (
                services_stack.enter_context(NoveltyService(
                    os.path.join(broker_dir, "novelty.sock"),
                    python=sys.executable,
                ))
                if needs_novelty
                else None
            )
            boltz_socket_by_gpu = {}
            for service_index, gpu_id in enumerate(boltz_worker_gpus):
                if not needs_boltz:
                    break
                socket_path = os.path.join(service_dir, f"boltz_gpu_{gpu_id}.sock")
                jit_cache_dir = os.path.join(service_dir, f"jit_gpu_{gpu_id}")
                log_path = os.path.join(
                    run_results_dir,
                    f"boltz_worker_{service_index}_gpu_{gpu_id}.log",
                )
                logger.info(
                    "Starting shared lazy persistent evaluator Boltz worker %d on GPU %d; log=%s",
                    service_index,
                    gpu_id,
                    log_path,
                )
                boltz_socket = services_stack.enter_context(BoltzService(
                    socket_path=socket_path,
                    cache=boltz_cache,
                    gpu_id=gpu_id,
                    log_path=log_path,
                    python=args.oracle_python,
                    boltz_executable=args.boltz_executable,
                    threads=args.boltz_cpu_threads,
                    startup_timeout=args.boltz_worker_startup_timeout,
                    jit_cache_dir=jit_cache_dir,
                    lazy=True,
                ))
                boltz_socket_by_gpu[gpu_id] = boltz_socket

            tasks = queue.Queue()
            for task_id in pending_task_ids:
                tasks.put(task_id)

            def worker(gpu_id, boltz_socket):
                while True:
                    try:
                        task_id = tasks.get_nowait()
                    except queue.Empty:
                        return
                    try:
                        result = run_task(
                            task_id,
                            args,
                            run_results_dir,
                            runtime_root,
                            msa_socket,
                            novelty_socket,
                            boltz_socket,
                            gpu_id,
                        )
                    except Exception as exc:
                        result = _error_result(task_id, f"{type(exc).__name__}: {exc}")
                        _write_result(os.path.join(run_results_dir, task_id), result)
                    with lock:
                        counters["done"] += 1
                        counters["errored"] += result.get("status") == "errored"
                        logger.info(
                            "[%d/%d] %s -> %s",
                            counters["done"],
                            len(pending_task_ids),
                            task_id,
                            result.get("status"),
                        )
                    tasks.task_done()

            threads = [
                threading.Thread(
                    target=worker,
                    args=(gpu, boltz_socket_by_gpu.get(gpu)),
                    daemon=False,
                )
                for gpu in worker_devices
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        summary = build_summary(
            args.run_name,
            args.results_dir,
            is_final=agent_run_is_complete(args.agent_outputs),
            task_ids=task_ids,
        )
        logger.info("Summarized %d tasks in %.0fs; metrics executed=%d; passed=%d; errored=%d; summary=%s",
                    summary["tasks_evaluated"], time.monotonic() - started,
                    summary["tasks_metrics_executed"],
                    summary["tasks_passed"], summary["tasks_errored"],
                    os.path.join(run_results_dir, "summary.json"))
        if not summary["is_final"]:
            logger.info("Agent run is still in progress; final_score remains null")
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)


if __name__ == "__main__":
    main()
