import argparse
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem

from agent_harness.config import AgentConfig
from agent_harness.prompts import TOOL_SCHEMAS, build_system_prompt
from agent_harness.run import (
    add_agent_config_arguments,
    apply_agent_config_arguments,
    remove_error_task_output,
)
from agent_harness.tools import ToolExecutor
from smdd_runtime.validate_molecule import validate_molecule


class MoleculeValidationTests(unittest.TestCase):
    def test_fixed_validator_returns_descriptors_alerts_and_similarity(self):
        with tempfile.TemporaryDirectory() as root:
            reference_path = str(Path(root, "reference.sdf"))
            writer = Chem.SDWriter(reference_path)
            writer.write(Chem.MolFromSmiles("CCO"))
            writer.close()

            result = validate_molecule("CCO", reference_path)

        self.assertTrue(result["valid"])
        self.assertTrue(result["sanitizable"])
        self.assertEqual(result["fragment_count"], 1)
        self.assertEqual(result["radical_electrons"], 0)
        self.assertAlmostEqual(
            result["reference_similarity"]["tanimoto_morgan_radius_2_2048"],
            1.0,
        )
        self.assertEqual(result["alerts"], {"pains": [], "brenk": [], "nih": []})
        self.assertIn("molecular_weight", result["descriptors"])
        self.assertTrue(result["default_hard_constraints_pass"])

    def test_fixed_validator_rejects_invalid_smiles(self):
        result = validate_molecule("not a smiles", None)
        self.assertFalse(result["valid"])
        self.assertFalse(result["sanitizable"])
        self.assertIn("error", result)

    def test_validation_tool_is_exposed_to_the_model(self):
        names = [schema["name"] for schema in TOOL_SCHEMAS]
        self.assertIn("validate_molecule", names)


class BoltzRepeatLimitTests(unittest.TestCase):
    @staticmethod
    def _arguments(smiles="CCO", num_samples=1):
        return {
            "reasoning": "test",
            "smiles": smiles,
            "protein_sequences": [{"id": "A", "sequence": "ACDEFG"}],
            "pocket_residues": [["A", 2]],
            "num_samples": num_samples,
        }

    def test_num_samples_and_equivalent_smiles_do_not_bypass_signature(self):
        executor = ToolExecutor(None, AgentConfig())
        first = executor._boltz_request_signature(self._arguments("C(C)O", 1))
        second = executor._boltz_request_signature(self._arguments("CCO", 5))
        self.assertEqual(first, second)

    def test_third_identical_request_is_rejected_without_spending_budget(self):
        config = AgentConfig(boltz_budget=8, boltz_max_repeats_per_request=2)
        executor = ToolExecutor(None, config)
        arguments = self._arguments()
        signature = executor._boltz_request_signature(arguments)
        executor._boltz_request_calls[signature] = 2
        executor.boltz_calls = 2

        output, metadata = executor.run_predict_boltz(arguments)

        self.assertIn("Repeated Boltz request limit reached", output)
        self.assertTrue(metadata["repeat_limit_exceeded"])
        self.assertEqual(metadata["request_calls"], 2)
        self.assertEqual(metadata["budget_remaining"], 6)
        self.assertEqual(executor.boltz_calls, 2)

    def test_repeat_limit_is_configurable_without_limiting_model_output(self):
        parser = argparse.ArgumentParser()
        add_agent_config_arguments(parser)
        args = parser.parse_args(["--boltz-max-repeats-per-request", "3"])
        config = AgentConfig()
        apply_agent_config_arguments(config, args)

        self.assertEqual(config.boltz_max_repeats_per_request, 3)
        self.assertIsNone(config.max_output_tokens)
        self.assertIn("at most 3 times", build_system_prompt(config))


class ErrorOutputCleanupTests(unittest.TestCase):
    def test_error_status_removes_only_its_task_directory(self):
        with tempfile.TemporaryDirectory() as root:
            error_dir = Path(root, "task_error")
            submitted_dir = Path(root, "task_submitted")
            error_dir.mkdir()
            submitted_dir.mkdir()
            Path(error_dir, "agent_result.json").write_text("{}", encoding="utf-8")
            Path(submitted_dir, "agent_result.json").write_text("{}", encoding="utf-8")

            self.assertTrue(remove_error_task_output(root, "task_error", "error"))
            self.assertFalse(error_dir.exists())
            self.assertTrue(submitted_dir.exists())

    def test_non_error_status_keeps_task_directory(self):
        with tempfile.TemporaryDirectory() as root:
            task_dir = Path(root, "task_submitted")
            task_dir.mkdir()

            self.assertFalse(
                remove_error_task_output(root, "task_submitted", "submitted")
            )
            self.assertTrue(task_dir.exists())

    def test_cleanup_rejects_paths_outside_output_root(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                remove_error_task_output(root, "../outside", "error")


if __name__ == "__main__":
    unittest.main()
