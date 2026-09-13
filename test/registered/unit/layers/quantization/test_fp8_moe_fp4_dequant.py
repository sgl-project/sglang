"""Regression tests for the ROCm AITER FP4-to-FP8 MoE wire-up."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization.fp8 as fp8
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_fp4_layer(num_experts: int = 2) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros((num_experts, 128, 64), dtype=torch.int8), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros((num_experts, 128, 64), dtype=torch.int8), requires_grad=False
    )
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.ones((num_experts, 128, 4), dtype=torch.float32), requires_grad=False
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.ones((num_experts, 128, 4), dtype=torch.float32), requires_grad=False
    )
    layer.w13_weight_scale_inv.format_ue8m0 = True
    layer.w2_weight_scale_inv.format_ue8m0 = True
    layer.w13_input_scale = None
    layer.w2_input_scale = None
    return layer


def _make_method(runner_is_aiter: bool) -> fp8.Fp8MoEMethod:
    method = fp8.Fp8MoEMethod.__new__(fp8.Fp8MoEMethod)
    method.is_fp4_expert = True
    method.dequant_fp4_to_fp8 = True
    method.weight_block_size = [32, 32]
    method.runner = SimpleNamespace(
        runner_backend=SimpleNamespace(is_aiter=lambda: runner_is_aiter)
    )
    return method


class TestFp8MoeFp4Dequant(unittest.TestCase):
    def test_aiter_dequantizes_fp4_experts_and_shuffles(self):
        layer = _make_fp4_layer()
        method = _make_method(runner_is_aiter=True)

        with (
            patch.object(fp8, "_use_aiter", True),
            patch.object(fp8, "_is_fp8_fnuz", False),
            patch.object(
                fp8,
                "shuffle_weight",
                side_effect=lambda weight, *_args: weight,
                create=True,
            ) as shuffle,
        ):
            method.process_weights_after_loading_block_quant(layer)

        self.assertFalse(method.is_fp4_expert)
        self.assertEqual(method.weight_block_size, [128, 128])
        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w2_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w13_weight_scale_inv.dtype, torch.float32)
        self.assertEqual(layer.w2_weight_scale_inv.dtype, torch.float32)
        self.assertFalse(layer.w13_weight_scale_inv.format_ue8m0)
        self.assertFalse(layer.w2_weight_scale_inv.format_ue8m0)
        self.assertEqual(shuffle.call_count, 2)

    def test_triton_runner_keeps_dequantized_weights_unshuffled(self):
        layer = _make_fp4_layer()
        method = _make_method(runner_is_aiter=False)

        with (
            patch.object(fp8, "_use_aiter", True),
            patch.object(fp8, "_is_fp8_fnuz", False),
            patch.object(fp8, "shuffle_weight", create=True) as shuffle,
        ):
            method.process_weights_after_loading_block_quant(layer)

        self.assertFalse(method.is_fp4_expert)
        shuffle.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=3)
