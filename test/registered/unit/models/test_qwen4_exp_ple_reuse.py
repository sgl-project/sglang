"""Unit tests for the PLE backing-file rewrite skip.

Place at: test/registered/unit/models/test_qwen4_exp_ple_reuse.py
Run with: python -m unittest test.registered.unit.models.test_qwen4_exp_ple_reuse

CPU only: these exercise the byte-window comparison, not the gather kernel.
"""

import unittest

import torch

import sglang.srt.models.qwen4_exp as qwen4_exp
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# 64 rows x 160 cols of bf16 is 20,480 bytes, past the 2-window cutoff, so these
# go through the sampled path. The first and last windows are always sampled.
_ROWS = 64
_COLS = 160


def _table(fill: float = 1.0, dtype=torch.bfloat16) -> torch.Tensor:
    table = torch.full((_ROWS, _COLS), fill, dtype=dtype)
    # Vary the contents so a wrong-offset comparison cannot pass by accident.
    table += torch.arange(_ROWS, dtype=dtype).unsqueeze(1) / 128.0
    return table


class TestPleShardMatches(CustomTestCase):
    def test_identical_matches(self):
        dst, src = _table(), _table()
        self.assertTrue(qwen4_exp._ple_shard_matches(dst, src))

    def test_first_window_difference_detected(self):
        dst, src = _table(), _table()
        dst[0, 0] += 1.0
        self.assertFalse(qwen4_exp._ple_shard_matches(dst, src))

    def test_last_window_difference_detected(self):
        dst, src = _table(), _table()
        dst[_ROWS - 1, _COLS - 1] += 1.0
        self.assertFalse(qwen4_exp._ple_shard_matches(dst, src))

    def test_wholly_different_contents_detected(self):
        dst, src = _table(fill=1.0), _table(fill=2.0)
        self.assertFalse(qwen4_exp._ple_shard_matches(dst, src))

    def test_shape_mismatch_falls_back(self):
        dst = _table()
        src = _table()[: _ROWS // 2]
        self.assertFalse(qwen4_exp._ple_shard_matches(dst, src))

    def test_dtype_mismatch_falls_back(self):
        dst = _table(dtype=torch.bfloat16)
        src = _table(dtype=torch.float16)
        self.assertFalse(qwen4_exp._ple_shard_matches(dst, src))

    def test_empty_shard_matches(self):
        empty = torch.empty((0, _COLS), dtype=torch.bfloat16)
        self.assertTrue(qwen4_exp._ple_shard_matches(empty, empty.clone()))

    def test_disabled_by_env(self):
        dst, src = _table(), _table()
        with envs.SGLANG_QWEN4_PLE_FILE_REUSE.override(False):
            self.assertFalse(qwen4_exp._ple_shard_matches(dst, src))

    def test_small_shard_compared_in_full(self):
        # 8 x 160 bf16 is 2,560 bytes, under the two-window cutoff, so the
        # comparison is exact rather than sampled: any single byte is caught.
        small_dst = torch.ones((8, _COLS), dtype=torch.bfloat16)
        small_src = small_dst.clone()
        self.assertTrue(qwen4_exp._ple_shard_matches(small_dst, small_src))
        small_dst[4, 77] += 1.0
        self.assertFalse(qwen4_exp._ple_shard_matches(small_dst, small_src))

    def test_sampling_is_deterministic(self):
        # Same inputs must give the same verdict across calls: the generator is
        # seeded from the shard size, not from global RNG state.
        dst, src = _table(), _table()
        torch.manual_seed(0)
        first = qwen4_exp._ple_shard_matches(dst, src)
        torch.manual_seed(12345)
        second = qwen4_exp._ple_shard_matches(dst, src)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
