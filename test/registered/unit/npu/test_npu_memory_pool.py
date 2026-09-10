"""
Unit tests for sglang.srt.hardware_backend.npu.memory_pool_npu.
"""

import unittest

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    _init_npu_conv_state,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2, suite="stage-a-unit-test-npu")


class TestInitNpuConvState(unittest.TestCase):
    def _make_conv_state_in(self, layers=2, pool_size=4, dtype=torch.bfloat16):
        return torch.zeros((layers, pool_size), dtype=dtype, device="cpu")

    def test_basic_shape(self):
        conv_state_in = self._make_conv_state_in(layers=2, pool_size=8)
        conv_state_shape = [(3, 16), (5, 32)]  # (dim, conv_wind)
        result = _init_npu_conv_state(conv_state_in, conv_state_shape)
        self.assertEqual(len(result), 2)
        # Entry 0: (2, 8, 16, 3) — conv_wind + 0 extra, dim first
        self.assertEqual(result[0].shape, (2, 8, 16, 3))
        # Entry 1: (2, 8, 32, 5)
        self.assertEqual(result[1].shape, (2, 8, 32, 5))

    def test_dim_swap(self):
        conv_state_in = self._make_conv_state_in(layers=1, pool_size=1)
        result = _init_npu_conv_state(conv_state_in, [(7, 9)])
        # conv_shape = (7, 9): dim=7, wind=9 → output (1, 1, 9, 7)
        self.assertEqual(result[0].shape, (1, 1, 9, 7))

    def test_dtype_inherited(self):
        conv_state_in = torch.zeros((2, 4), dtype=torch.float16, device="cpu")
        result = _init_npu_conv_state(conv_state_in, [(8, 4)])
        self.assertEqual(result[0].dtype, torch.float16)

    def test_all_zeros(self):
        conv_state_in = self._make_conv_state_in()
        result = _init_npu_conv_state(conv_state_in, [(4, 4)])
        for t in result:
            self.assertTrue(torch.all(t == 0))

    def test_speculative_extends_conv_len(self):
        conv_state_in = self._make_conv_state_in(layers=1, pool_size=1)
        # conv_wind = 4, extra = 3-1 = 2 → total conv_wind = 6
        result = _init_npu_conv_state(
            conv_state_in, [(4, 4)], speculative_num_draft_tokens=3
        )
        self.assertEqual(result[0].shape, (1, 1, 6, 4))

    def test_speculative_none_no_extension(self):
        conv_state_in = self._make_conv_state_in(layers=1, pool_size=1)
        result = _init_npu_conv_state(
            conv_state_in, [(4, 4)], speculative_num_draft_tokens=None
        )
        self.assertEqual(result[0].shape, (1, 1, 4, 4))

    def test_multiple_conv_shapes(self):
        conv_state_in = self._make_conv_state_in(layers=3, pool_size=16)
        shapes = [(2, 4), (4, 8), (6, 16)]
        result = _init_npu_conv_state(conv_state_in, shapes)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, (3, 16, 4, 2))
        self.assertEqual(result[1].shape, (3, 16, 8, 4))
        self.assertEqual(result[2].shape, (3, 16, 16, 6))


if __name__ == "__main__":
    unittest.main()
