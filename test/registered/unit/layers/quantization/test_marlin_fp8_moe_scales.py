"""CPU regressions for Marlin FP8 scale repack and MoE block selection."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    marlin_fp8_scales_to_bf16,
    marlin_moe_fp8_block_scales,
    marlin_moe_mxfp8_scales,
    prepare_moe_fp8_layer_for_marlin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="weekly", runner_config="cpu")


class TestMarlinFp8MoeScales(CustomTestCase):
    def test_checkpoint_scales_match_marlin_groups(self):
        columns = torch.arange(300) // 128
        blocks = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        torch.testing.assert_close(
            marlin_moe_fp8_block_scales(blocks, size_n=300, block_n=128),
            blocks[:, columns].transpose(1, 2),
        )

        raw = torch.tensor([[[127, 126, 130], [0, 254, 120]]], dtype=torch.uint8)
        expected = torch.tensor([[[1.0, 2.0**-127], [0.5, 2.0**127], [8.0, 2.0**-7]]])
        for scales in (raw, raw.view(torch.float8_e8m0fnu)):
            torch.testing.assert_close(
                marlin_moe_mxfp8_scales(scales), expected, rtol=0, atol=0
            )

    def test_compensation_preserves_tiny_and_signed_scales(self):
        # Casting before compensation would round these FP32 subnormals to zero.
        scales = torch.tensor([2.0**-149, 2.0**-140, -0.25, 255.0])
        expected = torch.tensor(
            [2.0**-29, 2.0**-20, -(2.0**118), 255.0 * 2.0**120], dtype=torch.bfloat16
        )
        torch.testing.assert_close(
            marlin_fp8_scales_to_bf16(scales), expected, rtol=0, atol=0
        )
        for value, message in (
            (float("nan"), "finite"),
            (float("inf"), "finite"),
            (255.25, "range"),
            (-255.25, "range"),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                marlin_fp8_scales_to_bf16(torch.tensor([value]))

    def test_rejection_keeps_both_stages_untouched(self):
        """An invalid down-projection scale leaves both sets of weights intact."""
        experts, hidden, inter = 2, 256, 128
        w13 = torch.nn.Parameter(
            torch.zeros((experts, 2 * inter, hidden), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        w2 = torch.nn.Parameter(
            torch.zeros((experts, hidden, inter), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer = SimpleNamespace(
            w13_weight=w13,
            w2_weight=w2,
            hidden_size=hidden,
            intermediate_size_per_partition=inter,
            moe_runner_config=SimpleNamespace(
                is_gated=True, activation="silu", apply_router_weight_on_input=False
            ),
        )
        good = torch.full((experts, 1, 2 * inter), 0.01)
        overflow = torch.full((experts, 1, hidden), 2.0**8)
        with self.assertRaisesRegex(ValueError, "w2 scales exceed"):
            prepare_moe_fp8_layer_for_marlin(layer, w13_scales=good, w2_scales=overflow)
        self.assertIs(layer.w13_weight, w13)
        self.assertIs(layer.w2_weight, w2)
        self.assertFalse(hasattr(layer, "w13_weight_scale"))

    def test_layer_without_a_gate_needs_128_aligned_rows(self):
        """Without a gate, gate/up N is the intermediate size (192 here)."""
        experts, hidden, inter = 2, 256, 192
        layer = SimpleNamespace(
            w13_weight=torch.zeros((experts, inter, hidden), dtype=torch.float8_e4m3fn),
            w2_weight=torch.zeros((experts, hidden, inter), dtype=torch.float8_e4m3fn),
            hidden_size=hidden,
            intermediate_size_per_partition=inter,
            moe_runner_config=SimpleNamespace(
                is_gated=False, activation="relu2", apply_router_weight_on_input=False
            ),
        )
        with self.assertRaisesRegex(ValueError, "without a gate"):
            prepare_moe_fp8_layer_for_marlin(
                layer,
                w13_scales=torch.full((experts, 1, inter), 0.01),
                w2_scales=torch.full((experts, 1, hidden), 0.01),
            )


if __name__ == "__main__":
    unittest.main()
