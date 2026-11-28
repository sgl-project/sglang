"""Unit tests for the CUDA search space of the block-wise FP8/INT8 GEMM tuner."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "benchmark/kernels/quantization/tuning_block_wise_kernel.py"
)


def _load_tuner():
    spec = importlib.util.spec_from_file_location("_block_wise_tuner", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # the script forces the "spawn" start method at import; keep that out of the test process
    with (
        patch("multiprocessing.set_start_method"),
        patch.dict(sys.modules, {"_block_wise_tuner": module}),
    ):
        spec.loader.exec_module(module)
    return module


class TestBlockWiseTunerSearchSpace(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tuner = _load_tuner()

    def _usable_tiles(self, block_k):
        with patch.object(self.tuner, "_is_hip", False):
            space = self.tuner.get_configs_compute_bound(block_k)
        return sorted(
            {c["BLOCK_SIZE_K"] for c in space if block_k % c["BLOCK_SIZE_K"] == 0}
        )

    def test_default_recipes_keep_their_tiles(self):
        self.assertEqual(self._usable_tiles(128), [64, 128])
        self.assertEqual(self._usable_tiles(64), [64])

    def test_small_group_gets_a_tile_that_divides_it(self):
        # a [32, 32] recipe used to leave an empty search space (#50057)
        self.assertEqual(self._usable_tiles(32), [32])

    def test_group_smaller_than_any_tile_has_no_tiles(self):
        self.assertEqual(self.tuner.get_cuda_block_k_tiles(16), [])


if __name__ == "__main__":
    unittest.main()
