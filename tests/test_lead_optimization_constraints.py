import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metrics.lead_optimization import assess_hard_constraints


class LeadOptimizationConstraintTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.solution = self.root / "solution.smi"
        self.reference = self.root / "reference.smi"
        self.solution.write_text("CCO\n", encoding="utf-8")
        self.reference.write_text("CCO\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_qed_threshold_accepts_boundary(self):
        with mock.patch("metrics.lead_optimization.QED.qed", return_value=0.45):
            result = assess_hard_constraints(
                str(self.solution),
                str(self.reference),
                qed_threshold=0.45,
            )

        self.assertTrue(result["qed"]["passed"])
        self.assertEqual(result["qed"]["threshold"], ">= 0.45")

    def test_qed_threshold_rejects_lower_score(self):
        with mock.patch("metrics.lead_optimization.QED.qed", return_value=0.4499):
            result = assess_hard_constraints(
                str(self.solution),
                str(self.reference),
                qed_threshold=0.45,
            )

        self.assertFalse(result["qed"]["passed"])
        self.assertFalse(result["all_passed"])

    def test_qed_is_opt_in_for_other_task_types(self):
        result = assess_hard_constraints(str(self.solution), str(self.reference))

        self.assertNotIn("qed", result)


if __name__ == "__main__":
    unittest.main()
