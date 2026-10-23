"""Unit tests for srt/models/kimi_k3_rocm_quant and the K3 weight-merge guard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
from sglang.srt.layers.quantization.quark.schemes import (
    QuarkW4A4MXFP4,
    quark_w4a4_mxfp4,
)
from sglang.srt.models.kimi_k3 import _is_unquantized_mergeable
from sglang.srt.models.kimi_k3_rocm_quant import (
    _k3_channel_fp8_to_bf16,
    densify_quark_mxfp4_shared_experts,
)
from sglang.test.test_utils import CustomTestCase


def _per_channel_fp8(out_features: int, in_features: int):
    torch.manual_seed(0)
    dense = torch.randn(out_features, in_features, dtype=torch.float32)
    # Spread the per-channel magnitudes so a wrongly-broadcast scale cannot
    # coincidentally reproduce the right answer.
    dense *= torch.logspace(-1, 1, out_features).unsqueeze(1)
    scale = dense.abs().amax(dim=1, keepdim=True) / 448.0
    weight = (dense / scale).to(torch.float8_e4m3fn)
    return weight, scale, dense


class TestChannelFp8ToBf16(CustomTestCase):
    """aiter's batched absorb GEMM takes ``w_scale`` as a scalar, so a
    per-channel vector cannot reach it. The scale axis is the GEMM's
    contraction axis, so requantizing to one per-tensor scale is lossy --
    ~2.3% relative error on real K3 weights. Dequantizing to bf16 costs ~0.12%
    and lets the absorb use its bf16 path with w_scale at the 1.0 default."""

    def test_dequantizes_every_channel_with_its_own_scale(self):
        weight, scale, dense = _per_channel_fp8(out_features=8, in_features=16)
        module = SimpleNamespace(weight_scale=scale.squeeze(1))

        out = _k3_channel_fp8_to_bf16(module, weight)

        self.assertEqual(out.dtype, torch.bfloat16)
        # Only the original fp8 rounding plus the bf16 cast separate this from
        # the checkpoint's own values; no second quantization.
        exact = weight.to(torch.float32) * scale
        torch.testing.assert_close(out.float(), exact, rtol=0.01, atol=0.0)
        # Guard the failure this replaced: one shared scale would leave the
        # low-magnitude channels far from their true values.
        worst = ((out.float() - dense).abs().sum(1) / dense.abs().sum(1)).max()
        self.assertLess(worst.item(), 0.05)

    def test_accepts_a_column_vector_scale(self):
        """Quark serializes the scale as [out]; callers may hold [out, 1]."""
        weight, scale, _ = _per_channel_fp8(out_features=8, in_features=16)

        flat = _k3_channel_fp8_to_bf16(
            SimpleNamespace(weight_scale=scale.squeeze(1)), weight
        )
        column = _k3_channel_fp8_to_bf16(
            SimpleNamespace(weight_scale=scale.clone()), weight
        )

        torch.testing.assert_close(flat, column)


class TestMergeDtypeGuard(CustomTestCase):
    """``_merge_weights_as_views`` cats ``.weight`` alone, so fusing a quantized
    projection silently drops the scale tensors that give its integer payload
    meaning. Only dtypes that carry their full value in ``.weight`` may fuse."""

    def test_rejects_weights_that_carry_a_separate_scale(self):
        for dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8):
            with self.subTest(dtype=dtype):
                weights = [torch.zeros(4, 4).to(dtype) for _ in range(2)]
                self.assertFalse(_is_unquantized_mergeable(weights))

    def test_accepts_plain_float_weights(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                weights = [torch.zeros(4, 4, dtype=dtype) for _ in range(2)]
                self.assertTrue(_is_unquantized_mergeable(weights))

    def test_rejects_mixed_dtypes(self):
        weights = [
            torch.zeros(4, 4, dtype=torch.bfloat16),
            torch.zeros(4, 4, dtype=torch.float16),
        ]
        self.assertFalse(_is_unquantized_mergeable(weights))

    def test_rejects_float32(self):
        """The merged buffer is only built for the two half dtypes the fused
        KDA/MoE kernels read; fp32 must not slip in through a dtype-agreement
        check that only asks whether the operands match each other."""
        weights = [torch.zeros(4, 4, dtype=torch.float32) for _ in range(2)]
        self.assertFalse(_is_unquantized_mergeable(weights))


def _quark_mxfp4_linear(out_features: int, in_features: int) -> torch.nn.Module:
    linear = torch.nn.Module()
    linear.quant_method = QuarkLinearMethod(quantization_config=None)
    linear.scheme = QuarkW4A4MXFP4(weight_quant_spec={}, input_quant_spec={})
    linear.scheme.create_weights(
        layer=linear,
        output_partition_sizes=[out_features],
        input_size_per_partition=in_features,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    # Byte 0x21 packs the codes for 0.5 (low nibble) and 1.0; scale byte 128 is 2^1.
    linear.weight.data.fill_(0x21)
    linear.weight_scale.data.fill_(128)
    return linear


class TestDensifyQuarkMxfp4SharedExperts(CustomTestCase):
    """Kimi-K3 densifies its MXFP4 shared experts before the loader's own
    post-processing, while other loaders post-process first. In either order
    each weight must be decoded exactly once into a mergeable BF16 weight."""

    def test_densify_and_loader_pass_commute(self):
        expected = torch.tensor([1.0, 2.0], dtype=torch.bfloat16).repeat(4, 16)
        for model_first in (True, False):
            with self.subTest(model_first=model_first):
                shared = SimpleNamespace(
                    gate_up_proj=_quark_mxfp4_linear(4, 32),
                    down_proj=_quark_mxfp4_linear(4, 32),
                )
                linears = (shared.gate_up_proj, shared.down_proj)
                # The loader pass densifies too when the HW lacks the FP4 GEMM.
                with patch.object(quark_w4a4_mxfp4, "_dequant_linear_to_bf16", True):
                    if model_first:
                        densify_quark_mxfp4_shared_experts(shared)
                    for linear in linears:
                        linear.quant_method.process_weights_after_loading(linear)
                    if not model_first:
                        densify_quark_mxfp4_shared_experts(shared)

                for linear in linears:
                    self.assertIsNone(linear.weight_scale)
                    torch.testing.assert_close(linear.weight, expected)
                self.assertTrue(
                    _is_unquantized_mergeable([linear.weight for linear in linears])
                )

    def test_asm_relaid_linear_is_left_alone(self):
        """Densify must keep a linear the loader re-laid out for the AITER ASM GEMM
        on that GEMM, not decode its shuffled weight as checkpoint MXFP4."""
        linear = _quark_mxfp4_linear(256, 256)
        with (
            patch.object(quark_w4a4_mxfp4, "_use_aiter_asm_fp4_gemm", True),
            patch.object(quark_w4a4_mxfp4, "is_gfx95_supported", return_value=True),
            patch.object(
                quark_w4a4_mxfp4,
                "shuffle_weight",
                side_effect=lambda weight, layout: weight.clone(),
                create=True,
            ),
        ):
            linear.quant_method.process_weights_after_loading(linear)
        self.assertTrue(linear.use_aiter_asm_fp4_gemm)
        weight, weight_scale = linear.weight, linear.weight_scale

        densify_quark_mxfp4_shared_experts(
            SimpleNamespace(gate_up_proj=linear, down_proj=linear)
        )

        self.assertFalse(linear.dequantized_bf16)
        self.assertIs(linear.weight, weight)
        self.assertIs(linear.weight_scale, weight_scale)


if __name__ == "__main__":
    unittest.main()
