import unittest

import torch
from triton.runtime.errors import OutOfResources

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ProgrammableLaunch:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.configs = []

    def __call__(self, config):
        self.configs.append(dict(config))
        outcome = self.outcomes.pop(0)
        if outcome is not None:
            raise outcome


class TestPersistentKernelFallback(CustomTestCase):
    def setUp(self):
        batch_invariant_ops._PERSISTENT_KERNEL_FALLBACK_CACHE.clear()
        self.config = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        }

    def tearDown(self):
        batch_invariant_ops._PERSISTENT_KERNEL_FALLBACK_CACHE.clear()

    @staticmethod
    def _shared_memory_oor():
        return OutOfResources(106496, 101376, "shared memory")

    def _launch(self, launcher, **overrides):
        params = {
            "launch": launcher,
            "operator": "matmul",
            "device": torch.device("cuda:0"),
            "dtype": torch.float16,
            "config": self.config,
        }
        params.update(overrides)
        batch_invariant_ops._launch_with_shared_memory_fallback(**params)

    def test_shared_memory_oor_retries_and_caches_successful_fallback(self):
        """A shared-memory launch failure falls back without changing its inputs."""
        original_config = dict(self.config)
        first_launch = _ProgrammableLaunch([self._shared_memory_oor(), None])

        with self.assertLogs(batch_invariant_ops.__name__, level="WARNING") as logs:
            self._launch(first_launch)

        self.assertEqual(
            [config["num_stages"] for config in first_launch.configs], [3, 2]
        )
        self.assertEqual(self.config, original_config)
        self.assertEqual(len(batch_invariant_ops._PERSISTENT_KERNEL_FALLBACK_CACHE), 1)
        self.assertIn("matmul", logs.output[0])
        self.assertIn("106496", logs.output[0])
        self.assertIn("101376", logs.output[0])

        cached_launch = _ProgrammableLaunch([None])
        with self.assertNoLogs(batch_invariant_ops.__name__, level="WARNING"):
            self._launch(cached_launch)
        self.assertEqual(cached_launch.configs[0]["num_stages"], 2)

    def test_non_shared_memory_oor_is_not_retried(self):
        error = OutOfResources(512, 255, "registers")
        launcher = _ProgrammableLaunch([error])

        with self.assertRaises(OutOfResources) as raised:
            self._launch(launcher)

        self.assertIs(raised.exception, error)
        self.assertEqual(len(launcher.configs), 1)
        self.assertFalse(batch_invariant_ops._PERSISTENT_KERNEL_FALLBACK_CACHE)

    def test_stage_two_shared_memory_oor_is_not_retried(self):
        self.config["num_stages"] = 2
        error = self._shared_memory_oor()
        launcher = _ProgrammableLaunch([error])

        with self.assertRaises(OutOfResources) as raised:
            self._launch(launcher)

        self.assertIs(raised.exception, error)
        self.assertEqual(len(launcher.configs), 1)
        self.assertFalse(batch_invariant_ops._PERSISTENT_KERNEL_FALLBACK_CACHE)

    def test_failed_fallback_is_not_cached(self):
        fallback_error = RuntimeError("fallback launch failed")
        launcher = _ProgrammableLaunch([self._shared_memory_oor(), fallback_error])

        with self.assertRaisesRegex(RuntimeError, "fallback launch failed"):
            self._launch(launcher)

        self.assertFalse(batch_invariant_ops._PERSISTENT_KERNEL_FALLBACK_CACHE)
        retry = _ProgrammableLaunch([None])
        self._launch(retry)
        self.assertEqual(retry.configs[0]["num_stages"], 3)

    def test_cache_key_isolates_launch_context(self):
        seed = _ProgrammableLaunch([self._shared_memory_oor(), None])
        with self.assertLogs(batch_invariant_ops.__name__, level="WARNING"):
            self._launch(seed)

        variants = [
            {"operator": "bmm"},
            {"device": torch.device("cuda:1")},
            {"dtype": torch.float32},
            {"config": {**self.config, "BLOCK_SIZE_N": 128}},
        ]
        for overrides in variants:
            with self.subTest(overrides=overrides):
                launcher = _ProgrammableLaunch([None])
                self._launch(launcher, **overrides)
                self.assertEqual(launcher.configs[0]["num_stages"], 3)


if __name__ == "__main__":
    unittest.main()
