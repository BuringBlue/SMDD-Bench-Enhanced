import os
import sys
import tempfile
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest import mock

from agent_harness.config import AgentConfig
from smdd_runtime.boltz_worker import _ResidentModels
from smdd_runtime.msa import DEFAULT_MSA_CACHE_DIR, PersistentMsaCache
from smdd_runtime.preflight import (
    check_model_api,
    validate_agent_config,
    validate_gpu_ids,
)


class StartupPreflightTests(unittest.TestCase):
    def test_gpu_selection_must_be_visible_and_unique(self):
        self.assertEqual(validate_gpu_ids([0, 2], [0, 1, 2]), [0, 2])
        with self.assertRaisesRegex(RuntimeError, "not visible"):
            validate_gpu_ids([3], [0, 1, 2])
        with self.assertRaisesRegex(RuntimeError, "must not be repeated"):
            validate_gpu_ids([0, 0], [0, 1])

    def test_unbounded_llm_timeout_is_rejected(self):
        config = replace(AgentConfig(), llm_timeout=50_000, task_timeout=43_200)
        with self.assertRaisesRegex(RuntimeError, "exceeds the per-task timeout"):
            validate_agent_config(config)

    def test_model_api_check_is_minimal_and_has_no_retries(self):
        completion = mock.Mock(return_value=object())
        fake_litellm = types.SimpleNamespace(completion=completion)
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            check_model_api("openai/test", False, 30)

        kwargs = completion.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 1)
        self.assertEqual(kwargs["num_retries"], 0)
        self.assertEqual(kwargs["timeout"], 30)

    def test_model_api_error_redacts_environment_secrets(self):
        secret = "test-secret-that-must-not-appear"
        completion = mock.Mock(side_effect=RuntimeError(f"bad key {secret}"))
        fake_litellm = types.SimpleNamespace(completion=completion)
        with (
            mock.patch.dict(sys.modules, {"litellm": fake_litellm}),
            mock.patch.dict(os.environ, {"TEST_API_KEY": secret}),
            self.assertRaises(RuntimeError) as raised,
        ):
            check_model_api("openai/test", False, 30)
        self.assertNotIn(secret, str(raised.exception))


class PersistentMsaCacheTests(unittest.TestCase):
    def test_cache_survives_new_cache_instance(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch(
                "smdd_runtime.msa._compute_uncached",
                return_value={"ACDE": "msa-content"},
            ) as compute:
                first = PersistentMsaCache(root).get_or_compute([" acde\n"])
                second = PersistentMsaCache(root).get_or_compute(["ACDE"])

        self.assertEqual(first, {"ACDE": "msa-content"})
        self.assertEqual(second, first)
        compute.assert_called_once_with(["ACDE"])

    def test_multichain_request_is_cached_as_one_paired_unit(self):
        with tempfile.TemporaryDirectory() as root:
            cache = PersistentMsaCache(root)
            with mock.patch(
                "smdd_runtime.msa._compute_uncached",
                side_effect=[
                    {"ACDE": "first", "FGHI": "second"},
                    {"FGHI": "second-new", "ACDE": "first-new"},
                ],
            ) as compute:
                first = cache.get_or_compute(["ACDE", "FGHI"])
                cached = cache.get_or_compute(["ACDE", "FGHI"])
                reordered = cache.get_or_compute(["FGHI", "ACDE"])

        self.assertEqual(first, cached)
        self.assertEqual(reordered, {"FGHI": "second-new", "ACDE": "first-new"})
        self.assertEqual(compute.call_count, 2)

    def test_default_cache_is_not_inside_task_visible_boltz_cache(self):
        self.assertFalse(
            os.path.commonpath((DEFAULT_MSA_CACHE_DIR, "/root/.boltz"))
            == "/root/.boltz"
        )

    def test_concurrent_identical_requests_compute_once(self):
        with tempfile.TemporaryDirectory() as root:
            cache = PersistentMsaCache(root)

            def slow_compute(_):
                time.sleep(0.05)
                return {"ACDE": "shared"}

            with mock.patch(
                "smdd_runtime.msa._compute_uncached",
                side_effect=slow_compute,
            ) as compute, ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(cache.get_or_compute, [["ACDE"], ["ACDE"]]))

        self.assertEqual(results, [{"ACDE": "shared"}, {"ACDE": "shared"}])
        self.assertEqual(compute.call_count, 1)


class _FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    @staticmethod
    def synchronize():
        return None

    @staticmethod
    def empty_cache():
        return None


class _FakeTorch:
    cuda = _FakeCuda()


class _FakeModel:
    def __init__(self, oom=False):
        self.oom = oom
        self.cpu_calls = 0
        self.cuda_calls = 0
        self._trainer = object()

    def eval(self):
        return self

    def cuda(self):
        self.cuda_calls += 1
        if self.oom:
            raise _FakeCuda.OutOfMemoryError("CUDA out of memory")
        return self

    def cpu(self):
        self.cpu_calls += 1
        return self


class GpuResidencyTests(unittest.TestCase):
    @staticmethod
    def _resident(models):
        resident = _ResidentModels.__new__(_ResidentModels)
        resident.models = {str(index): model for index, model in enumerate(models)}
        resident.active_model = models[-1]
        resident.low_memory_mode = False
        resident.torch = _FakeTorch()
        return resident

    def test_successful_models_are_retained_on_gpu(self):
        models = [_FakeModel(), _FakeModel()]
        resident = self._resident(models)
        resident._retain_on_gpu()
        self.assertFalse(resident.low_memory_mode)
        self.assertEqual([model.cuda_calls for model in models], [1, 1])
        self.assertEqual([model.cpu_calls for model in models], [0, 0])

    def test_residency_oom_falls_back_to_cpu(self):
        models = [_FakeModel(), _FakeModel(oom=True)]
        resident = self._resident(models)
        resident._retain_on_gpu()
        self.assertTrue(resident.low_memory_mode)
        self.assertIsNone(resident.active_model)
        self.assertEqual([model.cpu_calls for model in models], [1, 1])


if __name__ == "__main__":
    unittest.main()
