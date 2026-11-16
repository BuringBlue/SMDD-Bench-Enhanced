import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from rdkit import Chem

from metrics.novelty import assess_novelty_task4
from smdd_runtime import novelty


def _novelty_result(similarity=0.8):
    return {
        "canonical_smiles": "CCO",
        "search_floor": 0.4,
        "database_similarity": {
            "chembl": {"max_similarity": similarity},
            "drugbank": {"max_similarity": None},
            "patent": {"max_similarity": 0.5},
        },
        "scaffold_novelty": {
            "scaffold_smiles": None,
            "applicable": False,
            "seen_in": [],
            "is_novel": None,
            "complete": True,
            "affects_pass": False,
        },
    }


class NoveltyMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.solution = self.root / "solution.smi"
        self.solution.write_text("CCO\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    @mock.patch("metrics.novelty.request_novelty")
    def test_database_and_scaffold_results_are_record_only_by_default(self, request):
        request.return_value = _novelty_result(similarity=0.99)

        result = assess_novelty_task4(str(self.solution))

        self.assertTrue(result["all_passed"])
        self.assertEqual(result["mode"], "record_only")
        self.assertFalse(result["scaffold_novelty"]["affects_pass"])

    @mock.patch("metrics.novelty.request_novelty")
    def test_optional_database_similarity_threshold_is_enforced(self, request):
        request.return_value = _novelty_result(similarity=0.8)

        result = assess_novelty_task4(
            str(self.solution),
            max_database_similarity=0.8,
        )

        self.assertFalse(result["all_passed"])
        self.assertEqual(result["observed_max_similarity"], 0.8)


class NoveltySourceTests(unittest.TestCase):
    @mock.patch("smdd_runtime.novelty._request_json")
    def test_chembl_response_is_normalized(self, request):
        request.side_effect = [
            {"molecules": [{"molecule_chembl_id": "CHEMBL1", "similarity": "87.5"}]},
            {"molecules": [{"molecule_chembl_id": "CHEMBL2"}]},
        ]

        result = novelty._chembl_query("CCO", "c1ccccc1", 0.4)

        self.assertEqual(result["max_similarity"], 0.875)
        self.assertEqual(result["match_id"], "CHEMBL1")
        self.assertTrue(result["scaffold_seen"])

    @mock.patch("smdd_runtime.novelty._request_json")
    def test_surechembl_response_is_normalized(self, request):
        request.side_effect = [
            {"chemical_ids": [[123, 91.0]]},
            {"chemical_ids": [[456, 100.0]]},
        ]

        result = novelty._surechembl_query("CCO", "c1ccccc1", 0.4)

        self.assertEqual(result["max_similarity"], 0.91)
        self.assertEqual(result["match_id"], "123")
        self.assertEqual(result["scaffold_match_id"], "456")

    def test_local_drugbank_sdf_is_queried_without_network(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "drugbank.sdf"
            molecule = Chem.MolFromSmiles("c1ccccc1O")
            molecule.SetProp("DRUGBANK_ID", "DBTEST")
            writer = Chem.SDWriter(str(path))
            writer.write(molecule)
            writer.close()

            result = novelty._drugbank_query(
                "Oc1ccccc1",
                Chem.MolFromSmiles("Oc1ccccc1"),
                Chem.MolFromSmiles("c1ccccc1"),
                0.4,
                str(path),
            )

        self.assertEqual(result["max_similarity"], 1.0)
        self.assertEqual(result["match_id"], "DBTEST")
        self.assertTrue(result["scaffold_seen"])
        self.assertTrue(result["exhaustive_above_search_floor"])

    def test_request_uses_local_broker_when_configured(self):
        expected = _novelty_result()
        with tempfile.TemporaryDirectory() as root:
            socket_path = str(Path(root) / "novelty.sock")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(socket_path)
                server.listen(1)

                def handle_one_request():
                    connection, _ = server.accept()
                    novelty._handle_connection(connection)

                thread = threading.Thread(target=handle_one_request)
                thread.start()
                with (
                    mock.patch.dict(os.environ, {"SMDD_NOVELTY_SOCKET": socket_path}),
                    mock.patch("smdd_runtime.novelty.compute_novelty", return_value=expected),
                ):
                    result = novelty.request_novelty("CCO", timeout=5)
                thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
