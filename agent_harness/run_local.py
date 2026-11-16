"""Local SMDD runner with one persistent Boltz service per physical GPU."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from contextlib import ExitStack
from dataclasses import asdict

from agent_harness.run import (
    add_agent_config_arguments,
    apply_agent_config_arguments,
    assign_worker_gpus,
    build_summary,
    detect_gpus,
    load_run_config,
    remove_error_task_output,
    resolve_tasks,
    task_is_complete,
)
from smdd_runtime.boltz import require_cache, stage_cache
from smdd_runtime.boltz_worker import BoltzService
from smdd_runtime.isolation import verify_isolation
from smdd_runtime.msa import MsaService, msa_cache_dir
from smdd_runtime.preflight import (
    check_model_api,
    validate_agent_config,
    validate_gpu_ids,
)
from smdd_runtime.run_state import write_agent_run_state


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smdd_agent_local")


def unique_gpu_ids(worker_gpus):
    """Return assigned GPU IDs once each, preserving assignment order."""
    return list(dict.fromkeys(worker_gpus))


def _write_failure(path, task_id, error):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "agent_result.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "task_id": task_id,
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
            "error": error,
        }, handle, indent=4)


class Supervisor:
    def __init__(self, timeout):
        self.timeout = timeout
        self.active = set()
        self.lock = threading.Lock()

    def stop_all(self):
        with self.lock:
            processes = list(self.active)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def run(self, command, env, log_path):
        with open(log_path, "w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            with self.lock:
                self.active.add(process)
            try:
                return process.wait(timeout=self.timeout), None
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                return -signal.SIGKILL, f"task timed out after {self.timeout} seconds"
            finally:
                with self.lock:
                    self.active.discard(process)


def main():
    parser = argparse.ArgumentParser(description="SMDD agent local sandbox backend")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tasks-dir", default="tasks")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--task-file")
    parser.add_argument("--model")
    add_agent_config_arguments(parser)
    parser.add_argument("--gpus")
    parser.add_argument("--workers", type=int,
                        help="Concurrent task workers; may exceed the selected GPU count")
    args = parser.parse_args()

    args.tasks_dir = os.path.abspath(args.tasks_dir)
    args.results_dir = os.path.abspath(args.results_dir)
    task_ids = resolve_tasks(args)

    run_results_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(run_results_dir, exist_ok=True)
    config_path = os.path.join(run_results_dir, "config.json")
    config = load_run_config(config_path)
    if args.model:
        config.model = args.model
    apply_agent_config_arguments(config, args)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=4)

    pending_tasks = [
        (index, task_id)
        for index, task_id in enumerate(task_ids)
        if not task_is_complete(run_results_dir, task_id, config.model)
    ]
    skipped = len(task_ids) - len(pending_tasks)
    logger.info("Selected=%d, already complete=%d, pending=%d",
                len(task_ids), skipped, len(pending_tasks))
    if not pending_tasks:
        build_summary(args.run_name, run_results_dir, config.model)
        write_agent_run_state(
            run_results_dir,
            "complete",
            run_name=args.run_name,
            model=config.model,
            selected_tasks=len(task_ids),
            pending_tasks=0,
        )
        logger.info("All selected tasks are complete; summary refreshed at %s",
                    os.path.join(run_results_dir, "agent_summary.json"))
        return

    write_agent_run_state(
        run_results_dir,
        "preflight",
        run_name=args.run_name,
        model=config.model,
        selected_tasks=len(task_ids),
        pending_tasks=len(pending_tasks),
    )

    try:
        validate_agent_config(config)
        detected_gpu_ids = detect_gpus()
        gpu_ids = (
            [int(value.strip()) for value in args.gpus.split(",")]
            if args.gpus
            else detected_gpu_ids
        )
        validate_gpu_ids(gpu_ids, detected_gpu_ids)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Startup preflight failed: {exc}") from exc

    workers = args.workers if args.workers is not None else len(gpu_ids)
    if workers < 1:
        raise SystemExit("--workers must be at least 1")

    oracle_python = os.environ.get("SMDD_ORACLE_PYTHON", "/opt/oracle/bin/python")
    if not os.path.isfile(oracle_python):
        raise SystemExit(f"Startup preflight failed: missing Oracle Python: {oracle_python}")
    try:
        shared_boltz_cache = require_cache()
    except RuntimeError as exc:
        raise SystemExit(f"Startup preflight failed: {exc}") from exc

    logger.info("Preflight: checking model API for %s", config.model)
    try:
        check_model_api(config.model, config.use_responses_api, config.llm_timeout)
    except RuntimeError as exc:
        raise SystemExit(f"Startup preflight failed: {exc}") from exc
    logger.info("Preflight: model API and GPU selection passed")

    write_agent_run_state(
        run_results_dir,
        "running",
        run_name=args.run_name,
        model=config.model,
        selected_tasks=len(task_ids),
        pending_tasks=len(pending_tasks),
    )

    worker_gpus = assign_worker_gpus(gpu_ids, workers)
    boltz_worker_gpus = unique_gpu_ids(worker_gpus)
    if workers > len(gpu_ids):
        logger.info(
            "%d agent workers share %d per-GPU Boltz workers (%s); "
            "Boltz requests targeting the same GPU will be serialized",
            workers,
            len(boltz_worker_gpus),
            boltz_worker_gpus,
        )

    with open(os.path.join(run_results_dir, "runtime.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "backend": "local-sandbox",
            "network": "none; MSA only through fixed Unix-socket broker",
            "msa_server": "https://api.colabfold.com",
            "msa_cache": msa_cache_dir(),
            "msa_cache_scope": "persistent-trusted-only",
            "boltz_max_parallel_samples": 2,
            "boltz_num_workers": 0,
            "boltz_mode": "persistent-gpu-resident-per-gpu",
            "boltz_oom_fallback": "cpu-backed-one-model-at-a-time-with-one-retry",
            "boltz_workers": len(boltz_worker_gpus),
            "boltz_worker_gpus": boltz_worker_gpus,
            "boltz_cpu_threads": config.boltz_cpu_threads,
            "gpus": gpu_ids,
            "workers": workers,
            "worker_gpus": worker_gpus,
            "unix_sockets": "denied in model code; allowed only in trusted Boltz GPU stage",
        }, handle, indent=4)

    runtime_dir = tempfile.mkdtemp(prefix="smdd_agent_run_")
    supervisor = Supervisor(config.task_timeout)
    try:
        denied_path = os.path.join(args.tasks_dir, pending_tasks[0][1], "task.yaml")
        verify_isolation(runtime_dir, denied_path)
        boltz_cache = stage_cache(shared_boltz_cache)
        if boltz_cache == shared_boltz_cache:
            logger.info("Using Boltz cache at %s", boltz_cache)
        else:
            logger.info("Staged Boltz cache at %s", boltz_cache)
        service = MsaService(
            os.path.join(runtime_dir, "msa.sock"),
            oracle_python,
            cache_dir=msa_cache_dir(),
        )
        with service as msa_socket, ExitStack() as boltz_stack:
            boltz_socket_by_gpu = {}
            for service_index, gpu_id in enumerate(boltz_worker_gpus):
                socket_path = os.path.join(runtime_dir, f"boltz_gpu_{gpu_id}.sock")
                log_path = os.path.join(
                    run_results_dir,
                    f"boltz_worker_{service_index}_gpu_{gpu_id}.log",
                )
                logger.info(
                    "Starting shared persistent Boltz worker %d on GPU %d; startup log=%s",
                    service_index,
                    gpu_id,
                    log_path,
                )
                boltz_socket = boltz_stack.enter_context(BoltzService(
                    socket_path=socket_path,
                    cache=boltz_cache,
                    gpu_id=gpu_id,
                    log_path=log_path,
                    python=oracle_python,
                    threads=config.boltz_cpu_threads,
                    startup_timeout=config.boltz_worker_startup_timeout,
                ))
                boltz_socket_by_gpu[gpu_id] = boltz_socket
                logger.info(
                    "Shared persistent Boltz worker %d for GPU %d is ready",
                    service_index,
                    gpu_id,
                )

            tasks = queue.Queue()
            for index, task_id in pending_tasks:
                tasks.put((index, task_id))

            counters = {"done": 0, "error": 0}
            counter_lock = threading.Lock()

            def worker(gpu_id, boltz_socket):
                while True:
                    try:
                        index, task_id = tasks.get_nowait()
                    except queue.Empty:
                        return
                    try:
                        result_dir = os.path.join(run_results_dir, task_id)
                        os.makedirs(result_dir, exist_ok=True)
                        command = [
                            sys.executable, "-m", "agent_harness.local_task",
                            "--task-id", task_id,
                            "--task-index", str(index),
                            "--tasks-dir", args.tasks_dir,
                            "--run-results-dir", run_results_dir,
                            "--config", config_path,
                            "--msa-socket", msa_socket,
                            "--boltz-socket", boltz_socket,
                        ]
                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                        try:
                            returncode, error = supervisor.run(command, env, os.path.join(result_dir, "task_process.log"))
                        except Exception as exc:
                            returncode, error = -1, f"{type(exc).__name__}: {exc}"
                        result_path = os.path.join(result_dir, "agent_result.json")
                        if error or not os.path.isfile(result_path):
                            _write_failure(result_dir, task_id, error or f"task process exited with code {returncode}")
                        try:
                            with open(result_path, "r", encoding="utf-8") as handle:
                                status = json.load(handle).get("status", "error")
                        except Exception as exc:
                            status = "error"
                            _write_failure(result_dir, task_id, f"Invalid agent result: {exc}")
                        with counter_lock:
                            counters["done"] += 1
                            counters["error"] += status == "error"
                            logger.info("[%d/%d] %s -> %s", counters["done"], len(pending_tasks), task_id, status)
                        remove_error_task_output(run_results_dir, task_id, status)
                    except Exception as exc:
                        result_dir = os.path.join(run_results_dir, task_id)
                        _write_failure(result_dir, task_id, f"Supervisor error: {type(exc).__name__}: {exc}")
                        with counter_lock:
                            counters["done"] += 1
                            counters["error"] += 1
                            logger.exception("[%d/%d] %s -> error", counters["done"], len(pending_tasks), task_id)
                        remove_error_task_output(run_results_dir, task_id, "error")
                    finally:
                        tasks.task_done()

            threads = [
                threading.Thread(
                    target=worker,
                    args=(gpu_id, boltz_socket_by_gpu[gpu_id]),
                    name=f"smdd-worker-{index}-gpu-{gpu_id}",
                    daemon=False,
                )
                for index, gpu_id in enumerate(worker_gpus)
            ]
            for thread in threads:
                thread.start()
            try:
                for thread in threads:
                    thread.join()
            except KeyboardInterrupt:
                supervisor.stop_all()
                for thread in threads:
                    thread.join()
                raise

        summary = build_summary(args.run_name, run_results_dir, config.model)
        write_agent_run_state(
            run_results_dir,
            "complete",
            run_name=args.run_name,
            model=config.model,
            selected_tasks=len(task_ids),
            pending_tasks=0,
        )
        logger.info("Completed %d tasks; submitted=%d; errors=%d; summary=%s",
                    summary["tasks_total"], summary["tasks_submitted"], counters["error"],
                    os.path.join(run_results_dir, "agent_summary.json"))
    finally:
        supervisor.stop_all()
        shutil.rmtree(runtime_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
