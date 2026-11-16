import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rdkit import Chem
from rdkit.Chem import AllChem

from evaluator.eval_entrypoint import EvaluationRunner
from metrics import docking


class DockingMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.solution = self.root / "solution.smi"
        self.solution.write_text("CCO\n", encoding="utf-8")
        self.receptor = self.root / "complex.pdb"
        self.receptor.write_text("ATOM\n", encoding="utf-8")
        self.reference = self.root / "reference.sdf"

        molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        self.assertEqual(AllChem.EmbedMolecule(molecule, params), 0)
        writer = Chem.SDWriter(str(self.reference))
        writer.write(molecule)
        writer.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    @mock.patch("metrics.docking._run_json_command")
    def test_vina_records_affinity_without_an_implicit_cutoff(self, run_command):
        run_command.return_value = {
            "best_affinity_kcal_mol": -7.5,
            "center": [1.0, 2.0, 3.0],
            "box_size": [18.0, 18.0, 18.0],
            "energies": [[-7.5]],
            "poses_sdf": "/tmp/poses.sdf",
        }
        result = docking.assess_vina_task4(
            str(self.solution),
            str(self.receptor),
            str(self.reference),
            str(self.root / "vina"),
        )
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["best_affinity_kcal_mol"], -7.5)
        self.assertIn("--center", run_command.call_args.args[0])

    @mock.patch("metrics.docking._run_json_command")
    def test_vina_optional_cutoff_is_enforced(self, run_command):
        run_command.return_value = {"best_affinity_kcal_mol": -6.0, "poses_sdf": None}
        result = docking.assess_vina_task4(
            str(self.solution),
            str(self.receptor),
            str(self.reference),
            str(self.root / "vina"),
            max_affinity_kcal_mol=-8.0,
        )
        self.assertFalse(result["all_passed"])

    @mock.patch("metrics.docking._run_json_command")
    def test_diffdock_records_ranked_poses(self, run_command):
        run_command.return_value = {
            "best_confidence": 0.25,
            "samples": 2,
            "inference_steps": 20,
            "actual_steps": 19,
            "poses": [
                {"rank": 1, "confidence": 0.25, "path": "/tmp/rank1_confidence0.25.sdf"},
                {"rank": 2, "confidence": 0.10, "path": "/tmp/rank2_confidence0.10.sdf"},
            ],
        }
        result = docking.assess_diffdock_task4(
            str(self.solution),
            str(self.receptor),
            str(self.root / "diffdock"),
            samples=2,
        )
        self.assertTrue(result["all_passed"])
        self.assertEqual(len(result["poses"]), 2)
        self.assertEqual(
            result["poses"][0]["path"],
            "diffdock/prediction/rank1_confidence0.25.sdf",
        )

    def test_results_input_resolves_under_results_directory(self):
        runner = EvaluationRunner.__new__(EvaluationRunner)
        runner.results_dir = str(self.root / "results")
        resolved = runner._resolve_input({"source": "results", "value": "vina"})
        self.assertEqual(resolved, str(self.root / "results" / "vina"))

    def test_requested_step_details_are_preserved(self):
        runner = EvaluationRunner.__new__(EvaluationRunner)
        runner.task_id = "details-test"
        runner.task_dir = str(self.root / "task")
        runner.output_dir = str(self.root / "output")
        runner.results_dir = str(self.root / "results")
        runner.task = {
            "evaluation": [{
                "name": "Docking",
                "condition": {"eq": True},
                "result_key": "all_passed",
                "include_details": True,
            }],
        }
        runner.logger = mock.Mock()
        runner._execute_step = mock.Mock(return_value={
            "all_passed": True,
            "best_affinity_kcal_mol": -7.5,
        })

        result = runner.run()

        self.assertEqual(
            result["steps"][0]["details"]["best_affinity_kcal_mol"],
            -7.5,
        )


if __name__ == "__main__":
    unittest.main()
