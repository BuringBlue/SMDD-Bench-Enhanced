import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent_env.oracle_tools.vina_dock import _parse_vina_results
from agent_harness.config import AgentConfig
from agent_harness.local_tools import LocalToolExecutor
from agent_harness.prompts import TOOL_SCHEMAS, build_system_prompt
from agent_harness.tools import ToolExecutor


class DockingToolSchemaTests(unittest.TestCase):
    def test_vina_and_diffdock_are_exposed(self):
        names = {schema["name"] for schema in TOOL_SCHEMAS}
        self.assertIn("dock_vina", names)
        self.assertIn("dock_diffdock", names)

    def test_budgets_are_described_in_system_prompt(self):
        prompt = build_system_prompt(AgentConfig(vina_budget=11, diffdock_budget=4))
        self.assertIn("Budget: 11 calls", prompt)
        self.assertIn("Budget: 4 calls", prompt)


class DockingToolExecutorTests(unittest.TestCase):
    def test_vina_gpu_pose_affinities_are_parsed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            poses = Path(temp_dir) / "poses.pdbqt"
            poses.write_text(
                "MODEL 1\nREMARK VINA RESULT: -8.4 0.000 0.000\nENDMDL\n"
                "MODEL 2\nREMARK VINA RESULT: -7.9 2.100 3.200\nENDMDL\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _parse_vina_results(poses),
                [[-8.4, 0.0, 0.0], [-7.9, 2.1, 3.2]],
            )

    def test_workspace_paths_reject_traversal_and_absolute_paths(self):
        executor = ToolExecutor("container", AgentConfig())
        with self.assertRaises(ValueError):
            executor._resolve_workspace_path("../secret.pdb", (".pdb",))
        with self.assertRaises(ValueError):
            executor._resolve_workspace_path("/etc/passwd", (".pdb",))

    def test_workspace_alias_maps_to_local_backend_path(self):
        executor = ToolExecutor(
            None,
            AgentConfig(),
            workspace_dir="/tmp/local-task-workspace",
        )
        path, relative = executor._resolve_workspace_path(
            "/workspace/receptor.pdb",
            (".pdb",),
        )
        self.assertEqual(path, "/tmp/local-task-workspace/receptor.pdb")
        self.assertEqual(relative, "receptor.pdb")

    def test_vina_dispatch_uses_gpu_backend(self):
        executor = ToolExecutor("container", AgentConfig(vina_budget=2))
        result = {
            "tool": "vina",
            "backend": "vina-gpu-2.1",
            "poses_pdbqt": "/workspace/vina_results/call_1/poses.pdbqt",
            "best_affinity_kcal_mol": -7.5,
        }
        with mock.patch.object(
            executor,
            "_docker_exec",
            side_effect=[("", 0), (json.dumps(result), 0)],
        ) as docker_exec:
            output, metadata = executor.run_dock_vina({
                "smiles": "CCO",
                "receptor_path": "receptor.pdb",
                "center": [1, 2, 3],
            })

        command = docker_exec.call_args_list[0].args[0]
        self.assertEqual(command[0], "/opt/vina/bin/python")
        self.assertIn("/opt/smdd-tools/vina_dock.py", command)
        self.assertIn("--search-depth", command)
        self.assertEqual(json.loads(output)["poses_pdbqt"], "vina_results/call_1/poses.pdbqt")
        self.assertEqual(metadata["budget_remaining"], 1)

    def test_diffdock_dispatch_uses_isolated_conda_prefix(self):
        executor = ToolExecutor("container", AgentConfig(diffdock_budget=2))
        result = {
            "tool": "diffdock",
            "poses": [{"rank": 1, "path": "/workspace/diffdock_results/call_1/prediction/rank1.sdf"}],
        }
        with mock.patch.object(
            executor,
            "_docker_exec",
            side_effect=[("", 0), (json.dumps(result), 0)],
        ) as docker_exec:
            output, metadata = executor.run_dock_diffdock({
                "smiles": "CCO",
                "receptor_path": "receptor.pdb",
            })

        command = docker_exec.call_args_list[0].args[0]
        self.assertEqual(command[0], "/opt/diffdock/bin/python")
        self.assertIn("/opt/smdd-tools/diffdock_dock.py", command)
        self.assertEqual(
            json.loads(output)["poses"][0]["path"],
            "diffdock_results/call_1/prediction/rank1.sdf",
        )
        self.assertEqual(metadata["budget_remaining"], 1)

    def test_budget_exhaustion_does_not_execute_container_command(self):
        executor = ToolExecutor("container", AgentConfig(vina_budget=0, diffdock_budget=0))
        with mock.patch.object(executor, "_docker_exec") as docker_exec:
            vina_output, _ = executor.run_dock_vina({})
            diffdock_output, _ = executor.run_dock_diffdock({})
        self.assertIn("budget exceeded", vina_output)
        self.assertIn("budget exceeded", diffdock_output)
        docker_exec.assert_not_called()

    def test_local_diffdock_uses_only_the_fixed_gpu_entrypoint(self):
        executor = LocalToolExecutor.__new__(LocalToolExecutor)
        executor.gpu_sandbox = mock.Mock()
        executor.gpu_sandbox.run.return_value = SimpleNamespace(
            stdout="ok",
            stderr="",
            returncode=0,
        )
        command = [
            "/opt/diffdock/bin/python",
            "/opt/smdd-tools/diffdock_dock.py",
            "--help",
        ]

        output, returncode = executor._exec_diffdock(command, 30)

        self.assertEqual((output, returncode), ("ok", 0))
        executor.gpu_sandbox.run.assert_called_once_with(command, 30)
        with self.assertRaises(ValueError):
            executor._exec_diffdock(["/bin/bash", "-c", "true"], 30)

    def test_local_vina_uses_only_the_fixed_gpu_entrypoint(self):
        executor = LocalToolExecutor.__new__(LocalToolExecutor)
        executor.gpu_sandbox = mock.Mock()
        executor.gpu_sandbox.run.return_value = SimpleNamespace(
            stdout="ok",
            stderr="",
            returncode=0,
        )
        command = [
            "/opt/vina/bin/python",
            "/opt/smdd-tools/vina_dock.py",
            "--help",
        ]

        output, returncode = executor._exec_vina(command, 30)

        self.assertEqual((output, returncode), ("ok", 0))
        executor.gpu_sandbox.run.assert_called_once_with(command, 30)
        with self.assertRaises(ValueError):
            executor._exec_vina(["/bin/bash", "-c", "true"], 30)


if __name__ == "__main__":
    unittest.main()
