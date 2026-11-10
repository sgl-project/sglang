"""CPU tests for SM70 Marlin MoE CTA-geometry stage selection."""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.moe.moe_wna16_marlin import (
    _DSV41_DECODE_MARLIN_VALUES,
    _sm70_marlin_moe_stage_values,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSm70MarlinMoeStage(CustomTestCase):
    def test_dsv41_flash_decode_shapes(self):
        w13 = _sm70_marlin_moe_stage_values(
            torch.float8_e8m0fnu, 8, 6, 1, 4608, 5120
        )
        w2 = _sm70_marlin_moe_stage_values(
            torch.float8_e8m0fnu, 8, 1, 6, 5120, 2304
        )
        self.assertEqual(w13[0], "dsv41_w13_decode")
        self.assertEqual(w2[0], "dsv41_w2_decode")
        self.assertEqual(w13[1], _DSV41_DECODE_MARLIN_VALUES)
        self.assertEqual(w2[1], _DSV41_DECODE_MARLIN_VALUES)

    def test_dsv41_prefill_is_unpinned(self):
        self.assertIsNone(
            _sm70_marlin_moe_stage_values(
                torch.float8_e8m0fnu, 64, 6, 2048, 4608, 5120
            )
        )

    def test_qwen38_decode_unchanged(self):
        w13 = _sm70_marlin_moe_stage_values(
            torch.float8_e4m3fn, 8, 10, 1, 320, 2560
        )
        w2 = _sm70_marlin_moe_stage_values(
            torch.float8_e4m3fn, 8, 1, 10, 2560, 160
        )
        self.assertEqual(w13[1], ("32x64x64x4x32x64x16", "1", "vector_words"))
        self.assertEqual(w2[1], ("64x256x32x4x64x64x32", "1", "lane_vectors"))


if __name__ == "__main__":
    unittest.main()
