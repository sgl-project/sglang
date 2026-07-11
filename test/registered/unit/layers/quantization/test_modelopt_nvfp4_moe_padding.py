"""CPU unit tests for ModelOpt NVFP4 MoE intermediate padding."""

import unittest

import torch

from sglang.srt.layers.quantization.modelopt_quant import (
    _pad_nvfp4_gated_moe_weights_for_swizzle,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _row_values(rows: int) -> torch.Tensor:
    return torch.arange(1, rows + 1, dtype=torch.int32).view(1, rows, 1)


def _column_values(columns: int) -> torch.Tensor:
    return torch.arange(1, columns + 1, dtype=torch.int32).view(1, 1, columns)


class TestNvfp4MoeIntermediatePadding(CustomTestCase):
    def test_gated_gemma4_tp2_pads_each_half_and_scales(self):
        # Gemma-4 has moe_intermediate_size=704. TP=2 produces 352 channels per
        # rank, while the fused gated w13 row layout must align to 128 overall.
        intermediate = 352
        w13_weight = _row_values(2 * intermediate)
        w13_weight_scale = _row_values(2 * intermediate) * 10
        w2_weight = _column_values(intermediate // 2)
        w2_weight_scale = _column_values(intermediate // 16)

        padded = _pad_nvfp4_gated_moe_weights_for_swizzle(
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
            group_size=16,
        )
        padded_w13, padded_w13_scale, padded_w2, padded_w2_scale = padded

        padded_intermediate = 384
        self.assertEqual(padded_w13.shape, (1, 2 * padded_intermediate, 1))
        self.assertEqual(padded_w13_scale.shape, (1, 2 * padded_intermediate, 1))
        self.assertEqual(padded_w2.shape, (1, 1, padded_intermediate // 2))
        self.assertEqual(padded_w2_scale.shape, (1, 1, padded_intermediate // 16))

        for source, result in (
            (w13_weight, padded_w13),
            (w13_weight_scale, padded_w13_scale),
        ):
            torch.testing.assert_close(
                result[:, :intermediate], source[:, :intermediate]
            )
            torch.testing.assert_close(
                result[:, padded_intermediate : padded_intermediate + intermediate],
                source[:, intermediate:],
            )
            self.assertEqual(
                torch.count_nonzero(result[:, intermediate:padded_intermediate]).item(),
                0,
            )
            self.assertEqual(torch.count_nonzero(result[:, -32:]).item(), 0)

        torch.testing.assert_close(padded_w2[..., : intermediate // 2], w2_weight)
        torch.testing.assert_close(
            padded_w2_scale[..., : intermediate // 16], w2_weight_scale
        )
        self.assertEqual(
            torch.count_nonzero(padded_w2[..., intermediate // 2 :]).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(padded_w2_scale[..., intermediate // 16 :]).item(),
            0,
        )

    def test_aligned_gated_layout_is_not_reallocated(self):
        intermediate = 384
        tensors = (
            torch.zeros(1, 2 * intermediate, 2, dtype=torch.uint8),
            torch.zeros(1, 2 * intermediate, 4, dtype=torch.float8_e4m3fn),
            torch.zeros(1, 8, intermediate // 2, dtype=torch.uint8),
            torch.zeros(1, 8, intermediate // 16, dtype=torch.float8_e4m3fn),
        )

        padded = _pad_nvfp4_gated_moe_weights_for_swizzle(*tensors, group_size=16)

        for source, result in zip(tensors, padded):
            self.assertIs(result, source)


if __name__ == "__main__":
    unittest.main()
