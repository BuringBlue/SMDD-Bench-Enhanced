import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

from evaluator.eval_entrypoint import EvaluationRunner
from evaluator.evaluate import (
    build_summary,
    filter_existing_agent_tasks,
    prepare_evaluation_tasks,
    task_is_complete,
)
from evaluator.evaluate_local import (
    assign_worker_gpus as assign_evaluator_worker_gpus,
    unique_gpu_ids as unique_evaluator_gpu_ids,
)
from metrics.boltz_manager import BoltzManager
from agent_harness.run_local import unique_gpu_ids
from smdd_runtime.boltz_worker import (
    BOLTZ_EXECUTABLE,
    BoltzService,
    _ResidentModels,
)
from smdd_runtime.run_state import agent_run_is_complete, write_agent_run_state


class FailFastTests(unittest.TestCase):
    def test_gate_failure_skips_only_marked_downstream_steps(self):
        module_name = "_smdd_test_metrics"
        module = types.ModuleType(module_name)
        module.fail = lambda: False
        calls = []
        module.following_gate = lambda: calls.append("following_gate") or True

        def unexpected_downstream():
            raise AssertionError("Short-circuited step must not run")

        module.unexpected_downstream = unexpected_downstream
        sys.modules[module_name] = module
        try:
            with tempfile.TemporaryDirectory() as root:
                task_dir = os.path.join(root, "task")
                output_dir = os.path.join(root, "output")
                results_dir = os.path.join(root, "results")
                for path in (task_dir, output_dir, results_dir):
                    os.makedirs(path)
                task = {
                    "id": "fail_fast_test",
                    "output_config": {"file_path": "solution.smi"},
                    "evaluation": [
                        {
                            "name": "First gate",
                            "module": module_name,
                            "function": "fail",
                            "inputs": [],
                            "condition": {"eq": True},
                            "short_circuit_on_failure": True,
                        },
                        {
                            "name": "Second gate",
                            "module": module_name,
                            "function": "following_gate",
                            "inputs": [],
                            "condition": {"eq": True},
                            "short_circuit_on_failure": True,
                        },
                        {
                            "name": "Boltz score",
                            "module": module_name,
                            "function": "unexpected_downstream",
                            "inputs": [],
                            "condition": {"eq": True},
                            "skip_if_prior_short_circuit": True,
                        },
                        {
                            "name": "Vina Docking",
                            "module": module_name,
                            "function": "unexpected_downstream",
                            "inputs": [],
                            "condition": {"eq": True},
                            "skip_if_prior_short_circuit": True,
                        },
                    ],
                }
                Path(task_dir, "task.yaml").write_text(
                    yaml.safe_dump(task), encoding="utf-8"
                )
                result = EvaluationRunner(task_dir, output_dir, results_dir).run()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(calls, ["following_gate"])
            self.assertEqual(
                [step["status"] for step in result["steps"]],
                ["failed", "passed", "skipped", "skipped"],
            )
            self.assertIn("First gate", result["steps"][2]["reason"])
        finally:
            sys.modules.pop(module_name, None)

    def test_downstream_failure_does_not_skip_later_steps(self):
        module_name = "_smdd_test_metrics"
        module = types.ModuleType(module_name)
        calls = []
        module.fail = lambda: calls.append("boltz") or False
        module.pass_metric = lambda: calls.append("dock") or True
        sys.modules[module_name] = module
        try:
            with tempfile.TemporaryDirectory() as root:
                task_dir = os.path.join(root, "task")
                output_dir = os.path.join(root, "output")
                results_dir = os.path.join(root, "results")
                for path in (task_dir, output_dir, results_dir):
                    os.makedirs(path)
                task = {
                    "id": "downstream_failure_test",
                    "output_config": {"file_path": "solution.smi"},
                    "evaluation": [
                        {
                            "name": "Boltz Binding & Affinity",
                            "module": module_name,
                            "function": "fail",
                            "inputs": [],
                            "condition": {"eq": True},
                            "skip_if_prior_short_circuit": True,
                        },
                        {
                            "name": "Vina Docking",
                            "module": module_name,
                            "function": "pass_metric",
                            "inputs": [],
                            "condition": {"eq": True},
                            "skip_if_prior_short_circuit": True,
                        },
                    ],
                }
                Path(task_dir, "task.yaml").write_text(
                    yaml.safe_dump(task), encoding="utf-8"
                )
                result = EvaluationRunner(task_dir, output_dir, results_dir).run()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(calls, ["boltz", "dock"])
            self.assertEqual(
                [step["status"] for step in result["steps"]],
                ["failed", "passed"],
            )
        finally:
            sys.modules.pop(module_name, None)


class StepResumeTests(unittest.TestCase):
    def test_errored_task_reuses_completed_steps(self):
        module_name = "_smdd_resume_metrics"
        module = types.ModuleType(module_name)
        calls = []

        def unexpected():
            raise AssertionError("Completed step must not run again")

        module.unexpected = unexpected
        module.retry = lambda: calls.append("retry") or True
        sys.modules[module_name] = module
        try:
            with tempfile.TemporaryDirectory() as root:
                task_dir = Path(root, "task")
                output_dir = Path(root, "output")
                results_dir = Path(root, "results")
                for path in (task_dir, output_dir, results_dir):
                    path.mkdir()
                task = {
                    "id": "resume_steps_test",
                    "output_config": {"file_path": "solution.smi"},
                    "evaluation": [
                        {
                            "name": "Completed pass",
                            "module": module_name,
                            "function": "unexpected",
                            "inputs": [],
                            "condition": {"eq": True},
                        },
                        {
                            "name": "Retry error",
                            "module": module_name,
                            "function": "retry",
                            "inputs": [],
                            "condition": {"eq": True},
                        },
                        {
                            "name": "Completed failure",
                            "module": module_name,
                            "function": "unexpected",
                            "inputs": [],
                            "condition": {"eq": True},
                        },
                    ],
                }
                (task_dir / "task.yaml").write_text(
                    yaml.safe_dump(task), encoding="utf-8"
                )
                (results_dir / "result.json").write_text(
                    json.dumps({
                        "task_id": "resume_steps_test",
                        "status": "errored",
                        "steps": [
                            {
                                "name": "Completed pass",
                                "status": "passed",
                                "result": True,
                                "condition": {"eq": True},
                                "duration_s": 1.0,
                            },
                            {
                                "name": "Retry error",
                                "status": "errored",
                                "error": "transient",
                                "duration_s": 1.0,
                            },
                            {
                                "name": "Completed failure",
                                "status": "failed",
                                "result": False,
                                "condition": {"eq": True},
                                "duration_s": 1.0,
                            },
                        ],
                    }),
                    encoding="utf-8",
                )
                (results_dir / "eval.log").write_text(
                    "previous evaluation log\n", encoding="utf-8"
                )

                result = EvaluationRunner(
                    str(task_dir), str(output_dir), str(results_dir)
                ).run()
                log_contents = (results_dir / "eval.log").read_text(
                    encoding="utf-8"
                )

            self.assertEqual(calls, ["retry"])
            self.assertIn("previous evaluation log", log_contents)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                [step["status"] for step in result["steps"]],
                ["passed", "passed", "failed"],
            )
            self.assertTrue(result["steps"][0]["reused"])
            self.assertNotIn("reused", result["steps"][1])
            self.assertTrue(result["steps"][2]["reused"])
        finally:
            sys.modules.pop(module_name, None)

    def test_skipped_step_runs_after_upstream_error_recovers(self):
        module_name = "_smdd_resume_metrics"
        module = types.ModuleType(module_name)
        calls = []
        module.pass_gate = lambda: calls.append("gate") or True
        module.pass_downstream = lambda: calls.append("downstream") or True
        sys.modules[module_name] = module
        try:
            with tempfile.TemporaryDirectory() as root:
                task_dir = Path(root, "task")
                output_dir = Path(root, "output")
                results_dir = Path(root, "results")
                for path in (task_dir, output_dir, results_dir):
                    path.mkdir()
                task = {
                    "id": "resume_skipped_test",
                    "output_config": {"file_path": "solution.smi"},
                    "evaluation": [
                        {
                            "name": "Recovered gate",
                            "module": module_name,
                            "function": "pass_gate",
                            "inputs": [],
                            "condition": {"eq": True},
                            "short_circuit_on_failure": True,
                        },
                        {
                            "name": "Previously skipped",
                            "module": module_name,
                            "function": "pass_downstream",
                            "inputs": [],
                            "condition": {"eq": True},
                            "skip_if_prior_short_circuit": True,
                        },
                    ],
                }
                (task_dir / "task.yaml").write_text(
                    yaml.safe_dump(task), encoding="utf-8"
                )
                (results_dir / "result.json").write_text(
                    json.dumps({
                        "task_id": "resume_skipped_test",
                        "status": "errored",
                        "steps": [
                            {"name": "Recovered gate", "status": "errored"},
                            {"name": "Previously skipped", "status": "skipped"},
                        ],
                    }),
                    encoding="utf-8",
                )

                result = EvaluationRunner(
                    str(task_dir), str(output_dir), str(results_dir)
                ).run()

            self.assertEqual(calls, ["gate", "downstream"])
            self.assertEqual(result["status"], "passed")
        finally:
            sys.modules.pop(module_name, None)


class ResumeTests(unittest.TestCase):
    def _write_result(self, root, task_id, result):
        task_dir = os.path.join(root, task_id)
        os.makedirs(task_dir, exist_ok=True)
        Path(task_dir, "result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

    def test_passed_and_failed_results_are_complete(self):
        with tempfile.TemporaryDirectory() as root:
            for status in ("passed", "failed"):
                task_id = f"task_{status}"
                self._write_result(root, task_id, {
                    "task_id": task_id,
                    "status": status,
                    "steps": [{"name": "metric", "status": status}],
                })
                self.assertTrue(task_is_complete(root, task_id))

    def test_errored_mismatched_and_corrupt_results_are_pending(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_result(root, "errored", {
                "task_id": "errored",
                "status": "errored",
                "steps": [],
            })
            self._write_result(root, "mismatch", {
                "task_id": "another-task",
                "status": "passed",
                "steps": [],
            })
            corrupt_dir = os.path.join(root, "corrupt")
            os.makedirs(corrupt_dir)
            Path(corrupt_dir, "result.json").write_text("{", encoding="utf-8")
            self.assertFalse(task_is_complete(root, "errored"))
            self.assertFalse(task_is_complete(root, "mismatch"))
            self.assertFalse(task_is_complete(root, "corrupt"))


class SubmissionGateTests(unittest.TestCase):
    @staticmethod
    def _write_agent_result(agent_root, task_id, status):
        task_dir = Path(agent_root, task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        Path(task_dir, "agent_result.json").write_text(
            json.dumps({"task_id": task_id, "status": status}),
            encoding="utf-8",
        )

    def test_only_submitted_tasks_are_returned_for_metric_execution(self):
        with tempfile.TemporaryDirectory() as root:
            agent_root = os.path.join(root, "agent")
            results_root = os.path.join(root, "results")
            run_name = "evaluation"
            run_results = os.path.join(results_root, run_name)
            submitted_id = "smdd_004_submitted_0"
            limited_id = "smdd_004_limited_0"
            missing_id = "smdd_004_missing_0"
            self._write_agent_result(agent_root, submitted_id, "submitted")
            self._write_agent_result(agent_root, limited_id, "max_turns_exceeded")

            submitted, not_submitted, incomplete = prepare_evaluation_tasks(
                [submitted_id, limited_id, missing_id],
                agent_root,
                run_results,
            )

            self.assertEqual(submitted, [submitted_id])
            self.assertEqual(not_submitted, {
                limited_id: "max_turns_exceeded",
            })
            self.assertEqual(incomplete, {missing_id: "missing_agent_result"})
            self.assertFalse(
                os.path.exists(os.path.join(run_results, submitted_id, "result.json"))
            )
            result = json.loads(
                Path(run_results, limited_id, "result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason_code"], "agent_not_submitted")
            self.assertFalse(result["metrics_executed"])
            self.assertFalse(
                os.path.exists(os.path.join(run_results, missing_id, "result.json"))
            )

            summary = build_summary(run_name, results_root, is_final=False)
            self.assertEqual(summary["tasks_evaluated"], 1)
            self.assertEqual(summary["tasks_failed"], 1)
            self.assertEqual(summary["tasks_not_submitted"], 1)
            self.assertEqual(summary["tasks_metrics_executed"], 0)
            self.assertFalse(summary["is_final"])
            self.assertIsNone(summary["final_score"])

    def test_later_submission_invalidates_synthetic_failure_for_resume(self):
        with tempfile.TemporaryDirectory() as root:
            agent_root = os.path.join(root, "agent")
            run_results = os.path.join(root, "evaluation")
            task_id = "smdd_004_resume_0"
            self._write_agent_result(agent_root, task_id, "max_turns_exceeded")
            prepare_evaluation_tasks([task_id], agent_root, run_results)
            self.assertFalse(task_is_complete(run_results, task_id))

            self._write_agent_result(agent_root, task_id, "submitted")
            submitted, not_submitted, incomplete = prepare_evaluation_tasks(
                [task_id], agent_root, run_results
            )
            self.assertEqual(submitted, [task_id])
            self.assertEqual(not_submitted, {})
            self.assertEqual(incomplete, {})
            self.assertFalse(task_is_complete(run_results, task_id))

    def test_every_selection_mode_is_limited_to_existing_agent_folders(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "task_a").mkdir()
            self.assertEqual(
                filter_existing_agent_tasks(["task_a", "task_b"], root),
                ["task_a"],
            )

    def test_agent_run_state_controls_final_score_readiness(self):
        with tempfile.TemporaryDirectory() as root:
            write_agent_run_state(
                root,
                "running",
                run_name="model/run/agent",
                model="model",
                selected_tasks=2,
                pending_tasks=2,
            )
            self.assertFalse(agent_run_is_complete(root))
            write_agent_run_state(
                root,
                "complete",
                run_name="model/run/agent",
                model="model",
                selected_tasks=2,
                pending_tasks=0,
            )
            self.assertTrue(agent_run_is_complete(root))

    def test_summary_is_limited_to_current_agent_task_folders(self):
        with tempfile.TemporaryDirectory() as root:
            run_name = "evaluation"
            run_root = Path(root, run_name)
            for task_id in ("smdd_004_current_0", "smdd_004_stale_0"):
                task_root = Path(run_root, task_id)
                task_root.mkdir(parents=True)
                Path(task_root, "result.json").write_text(
                    json.dumps({
                        "task_id": task_id,
                        "status": "failed",
                        "steps": [{"name": "metric", "status": "failed"}],
                    }),
                    encoding="utf-8",
                )
            summary = build_summary(
                run_name,
                root,
                task_ids=["smdd_004_current_0"],
            )
            self.assertEqual(summary["tasks_evaluated"], 1)
            self.assertEqual(list(summary["tasks"]), ["smdd_004_current_0"])

    def test_summary_counts_failed_evaluation_steps(self):
        with tempfile.TemporaryDirectory() as root:
            run_name = "evaluation"
            run_root = Path(root, run_name)
            results = {
                "smdd_004_first_0": {
                    "task_id": "smdd_004_first_0",
                    "status": "failed",
                    "steps": [
                        {"name": "Validity", "status": "passed"},
                        {"name": "Novelty", "status": "failed"},
                        {"name": "Vina Docking", "status": "failed"},
                    ],
                },
                "smdd_004_second_0": {
                    "task_id": "smdd_004_second_0",
                    "status": "failed",
                    "steps": [
                        {"name": "Novelty", "status": "failed"},
                        {"name": "Vina Docking", "status": "skipped"},
                    ],
                },
                "smdd_004_unsubmitted_0": {
                    "task_id": "smdd_004_unsubmitted_0",
                    "status": "failed",
                    "reason_code": "agent_not_submitted",
                    "metrics_executed": False,
                    "steps": [
                        {"name": "Agent submission", "status": "failed"},
                    ],
                },
            }
            for task_id, result in results.items():
                task_root = run_root / task_id
                task_root.mkdir(parents=True)
                (task_root / "result.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )

            summary = build_summary(run_name, root)

            self.assertEqual(summary["step_failure_counts"], {
                "Novelty": 2,
                "Vina Docking": 1,
            })


class PersistentWorkerTests(unittest.TestCase):
    def test_agent_workers_share_one_boltz_service_per_gpu(self):
        self.assertEqual(unique_gpu_ids([0, 0, 1, 0, 1]), [0, 1])

    def test_evaluator_workers_share_one_boltz_service_per_gpu(self):
        worker_gpus = assign_evaluator_worker_gpus([0], 4)
        self.assertEqual(worker_gpus, [0, 0, 0, 0])
        self.assertEqual(unique_evaluator_gpu_ids(worker_gpus), [0])

    def test_evaluator_worker_gpu_assignment_is_round_robin(self):
        self.assertEqual(
            assign_evaluator_worker_gpus([0, 1], 5),
            [0, 1, 0, 1, 0],
        )
        self.assertEqual(assign_evaluator_worker_gpus([], 3), [None, None, None])

    def test_validator_accepts_evaluator_msa_sibling_output(self):
        with tempfile.TemporaryDirectory() as root:
            msa_dir = os.path.join(root, ".msa")
            os.makedirs(msa_dir)
            input_path = os.path.join(msa_dir, "agent_dock.yaml")
            Path(input_path).write_text("version: 1\n", encoding="utf-8")
            output_path = os.path.join(root, "boltz_results")
            command = [
                BOLTZ_EXECUTABLE,
                "predict",
                input_path,
                "--out_dir",
                output_path,
                "--use_potentials",
                "--diffusion_samples",
                "10",
                "--seed",
                "42",
                "--diffusion_samples_affinity",
                "10",
                "--max_parallel_samples",
                "1",
                "--num_workers",
                "0",
            ]
            data_path, out_path = _ResidentModels._validate_command(command)
            self.assertEqual(data_path, os.path.realpath(input_path))
            self.assertEqual(out_path, os.path.realpath(output_path))

    def test_validator_accepts_pinned_evaluator_executable(self):
        with tempfile.TemporaryDirectory() as root:
            input_path = os.path.join(root, "agent_dock.yaml")
            Path(input_path).write_text("version: 1\n", encoding="utf-8")
            executable = "/opt/conda/bin/boltz"
            command = [
                executable,
                "predict",
                input_path,
                "--out_dir",
                os.path.join(root, "results"),
                "--use_potentials",
                "--diffusion_samples",
                "10",
                "--seed",
                "42",
                "--diffusion_samples_affinity",
                "10",
                "--max_parallel_samples",
                "1",
                "--num_workers",
                "0",
            ]
            with mock.patch.dict(
                os.environ, {"SMDD_BOLTZ_EXECUTABLE": executable}
            ):
                _ResidentModels._validate_command(command)

    def test_private_jit_environment_excludes_secrets(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": "must-not-leak"}
        ):
            service = BoltzService(
                socket_path=os.path.join(root, "worker.sock"),
                cache=root,
                gpu_id=0,
                log_path=os.path.join(root, "worker.log"),
                python=sys.executable,
                jit_cache_dir=os.path.join(root, "private-jit"),
                lazy=True,
            )
            env = service._worker_env()
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertEqual(
                env["SMDD_BOLTZ_EXECUTABLE"],
                os.path.join(os.path.dirname(sys.executable), "boltz"),
            )
            for key in (
                "TRITON_CACHE_DIR",
                "CUDA_CACHE_PATH",
                "TORCHINDUCTOR_CACHE_DIR",
                "CUEQ_TRITON_CACHE_DIR",
                "XDG_CACHE_HOME",
                "TMPDIR",
            ):
                self.assertTrue(env[key].startswith(service.jit_cache_dir + os.sep))
                self.assertEqual(os.stat(env[key]).st_mode & 0o777, 0o700)
            self.assertEqual(env["HOME"], service.jit_cache_dir)

    def test_manager_dispatches_to_worker(self):
        with tempfile.TemporaryDirectory() as root:
            manager = BoltzManager(root)
            yaml_path = manager.create_input_yaml([], "c1ccccc1")
            attached_dir = os.path.join(root, ".msa")
            os.makedirs(attached_dir)
            attached_yaml = os.path.join(attached_dir, "boltz_run.yaml")
            Path(attached_yaml).write_text("version: 1\n", encoding="utf-8")

            env = {
                "SMDD_MSA_SOCKET": "/private/msa.sock",
                "SMDD_BOLTZ_SOCKET": "/private/boltz.sock",
                "SMDD_BOLTZ_MAX_PARALLEL_SAMPLES": "1",
                "SMDD_BOLTZ_NUM_WORKERS": "0",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch("smdd_runtime.msa.attach_msas", return_value=attached_yaml),
                mock.patch(
                    "smdd_runtime.boltz_worker.request_boltz",
                    return_value=("worker output", 0),
                ) as request,
            ):
                manager.run_prediction(yaml_path, diffusion_samples=10)

            command = request.call_args.args[1]
            self.assertEqual(command[1], "predict")
            self.assertEqual(command[2], attached_yaml)
            self.assertIn("--diffusion_samples_affinity", command)


if __name__ == "__main__":
    unittest.main()
