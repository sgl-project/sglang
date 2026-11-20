"""Unit tests for the block-wise W8A8 GEMM tuner search space.

These cover the pure search-space construction only; the module under test has
no torch/triton dependency, so the logic runs on CPU.
"""

import unittest

from sglang.benchmark.tuning_configs import (
    get_configs_compute_bound,
    get_cuda_block_k_tiles,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCudaBlockKTiles(CustomTestCase):
    def test_default_search_space_is_unchanged(self):
        # The historical 64/128 K tiles must stay exactly as before.
        self.assertEqual(get_cuda_block_k_tiles(128), [64, 128])
        self.assertEqual(get_cuda_block_k_tiles(256), [64, 128])

    def test_small_quantization_block_is_supported(self):
        # Regression: a [32, 32] recipe used to filter every candidate out.
        self.assertEqual(get_cuda_block_k_tiles(32), [32])
        self.assertEqual(get_cuda_block_k_tiles(16), [16])

    def test_unsupported_block_size_raises(self):
        with self.assertRaises(ValueError):
            get_cuda_block_k_tiles(24)
        with self.assertRaises(ValueError):
            get_cuda_block_k_tiles(0)


class TestComputeBoundSearchSpace(CustomTestCase):
    def test_block_k_32_search_space_is_not_empty(self):
        configs = get_configs_compute_bound(is_hip=False, block_k=32)
        self.assertTrue(configs)
        filtered = [c for c in configs if 32 % c["BLOCK_SIZE_K"] == 0]
        self.assertTrue(filtered)
        self.assertEqual({c["BLOCK_SIZE_K"] for c in filtered}, {32})

    def test_every_cuda_candidate_divides_the_quantization_block(self):
        for block_k in (16, 32, 64, 128):
            with self.subTest(block_k=block_k):
                configs = get_configs_compute_bound(is_hip=False, block_k=block_k)
                self.assertTrue(configs)
                self.assertTrue(all(block_k % c["BLOCK_SIZE_K"] == 0 for c in configs))

    def test_rocm_path_ignores_block_k(self):
        configs = get_configs_compute_bound(is_hip=True, block_k=32)
        self.assertTrue(configs)
        self.assertIn("waves_per_eu", configs[0])


if __name__ == "__main__":
    unittest.main()
