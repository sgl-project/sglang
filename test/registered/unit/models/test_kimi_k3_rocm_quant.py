"""Unit tests for srt/models/kimi_k3_rocm_quant and the K3 weight-merge guard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8
from sglang.srt.models.kimi_k3 import _is_unquantized_mergeable
from sglang.srt.models.kimi_k3_rocm_quant import (
    _k3_channel_fp8_to_bf16,
    k3_kv_b_for_absorb,
    k3_merge_kda_inproj_fp8,
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


class TestKvBForAbsorb(CustomTestCase):
    """The MLA absorb BMMs take one scalar w_scale, so by default kv_b is
    requantized to per-tensor FP8; the kill switch keeps the BF16 dequant."""

    def test_default_requantizes_to_one_tensor_scale(self):
        weight, scale, _ = _per_channel_fp8(out_features=8, in_features=16)
        module = SimpleNamespace(weight_scale=scale.squeeze(1))

        q, tensor_scale = k3_kv_b_for_absorb(module=module, weight=weight)

        self.assertEqual((q.dtype, q.shape), (weight.dtype, weight.shape))
        self.assertEqual(tensor_scale.numel(), 1)
        # The largest channel spans the full FP8 range, so nothing clips.
        self.assertEqual(q.float().abs().max().item(), torch.finfo(q.dtype).max)
        # Error is bounded by one FP8 step at the tensor-wide scale.
        exact = weight.float() * scale
        err = (q.float() * tensor_scale - exact).abs().max()
        self.assertLess(err.item(), 0.07 * exact.abs().max().item())

    def test_kill_switch_keeps_the_bf16_dequant(self):
        weight, scale, _ = _per_channel_fp8(out_features=8, in_features=16)
        module = SimpleNamespace(weight_scale=scale.squeeze(1))

        with envs.SGLANG_OPT_HIP_K3_MLA_ABSORB_FP8.override(False):
            out, tensor_scale = k3_kv_b_for_absorb(module=module, weight=weight)

        self.assertIsNone(tensor_scale)
        torch.testing.assert_close(out, _k3_channel_fp8_to_bf16(module, weight))


def _quark_fp8_linear(out_features: int, in_features: int = 16):
    scheme = QuarkW8A8Fp8(
        {"qscheme": "per_channel"}, {"qscheme": "per_channel", "is_dynamic": True}
    )
    scheme.out_dtype = torch.bfloat16
    weight, scale, _ = _per_channel_fp8(out_features, in_features)
    return SimpleNamespace(
        weight=weight,
        weight_scale=scale.squeeze(1),
        scheme=scheme,
        quant_method=mock.Mock(spec=QuarkLinearMethod),
    )


def _dequant(linear) -> torch.Tensor:
    return linear.weight.float() * linear.weight_scale.view(-1, 1)


def _merge_kda_inproj(b_proj=None):
    return k3_merge_kda_inproj_fp8(
        qkvg_proj=_quark_fp8_linear(32),
        qkvg_sizes=[24, 8],
        f_a_proj=_quark_fp8_linear(8),
        b_proj=_quark_fp8_linear(3) if b_proj is None else b_proj,
        f_b_proj=_quark_fp8_linear(16, in_features=8),
    )


@mock.patch(
    "sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8."
    "use_aiter_bpreshuffle_gemm",
    return_value=False,
)
class TestMergeKdaInprojFp8(CustomTestCase):
    """The merged buffer has to carry every projection's channel scales; a raw
    row-cat of FP8 payloads would silently drop them."""

    def test_merged_linear_keeps_every_channel_scale(self, _):
        expected = torch.cat([_dequant(_quark_fp8_linear(n)) for n in (32, 8, 3)])

        layer, sizes, f_b = _merge_kda_inproj()

        # Processed like the Quark linear: [in, out] weight, [out, 1] scale.
        merged = layer.weight.t().float() * layer.weight_scale.view(-1, 1)
        # N % 64 == 0 keeps AITER's preshuffled FP8 GEMM.
        self.assertEqual(merged.shape[0] % 64, 0)
        torch.testing.assert_close(merged[:43], expected)
        self.assertEqual(merged[43:].abs().max().item(), 0.0)
        self.assertEqual(sizes, [24, 8, 8, 3, merged.shape[0] - 43])
        torch.testing.assert_close(
            f_b, _dequant(_quark_fp8_linear(16, in_features=8)).to(torch.bfloat16)
        )

    def test_skips_a_projection_that_is_not_quark_fp8(self, _):
        bf16_b_proj = SimpleNamespace(
            weight=torch.zeros(3, 16, dtype=torch.bfloat16), quant_method=None
        )
        self.assertIsNone(_merge_kda_inproj(b_proj=bf16_b_proj))

    def test_env_disables_the_merge(self, _):
        with envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ.override(False):
            self.assertIsNone(_merge_kda_inproj())


if __name__ == "__main__":
    unittest.main()
