"""Unit tests for srt/models/kimi_k3_rocm_quant, kimi_k3_rocm_moe and the K3
weight-merge guard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.models.kimi_k3 import _is_unquantized_mergeable
from sglang.srt.models.kimi_k3_rocm_moe import tuned_front_gemm, tuned_front_window
from sglang.srt.models.kimi_k3_rocm_quant import _k3_channel_fp8_to_bf16
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


def _shaped(rows: int, cols: int) -> torch.Tensor:
    # Only the shape is read; expand keeps the 6016x7168 front at one element.
    return torch.zeros(1, dtype=torch.bfloat16).expand(rows, cols)


class TestTunedFrontGemm(CustomTestCase):
    """AITER's tuned rows cover one front shape in bf16, so the tuned GEMM must take
    exactly its inclusive token window there and leave every other front alone."""

    def test_takes_only_the_inclusive_window_on_the_tuned_shape(self):
        tuned_out = torch.zeros(1)
        tuned_gemm = types.ModuleType("aiter.tuned_gemm")
        tuned_gemm.tgemm = SimpleNamespace(mm=lambda **kwargs: tuned_out)
        aiter = {"aiter": types.ModuleType("aiter"), "aiter.tuned_gemm": tuned_gemm}
        front = _shaped(6016, 7168)
        with patch.dict(sys.modules, aiter):
            for num_tokens, takes_tuned in (
                (47, False),
                (48, True),
                (192, True),
                (193, False),
            ):
                with self.subTest(num_tokens=num_tokens):
                    out = tuned_front_gemm(
                        x=_shaped(num_tokens, 7168),
                        weight=front,
                        out_dtype=None,
                        window=(48, 192),
                    )
                    self.assertIs(out is tuned_out, takes_tuned)
                    self.assertIs(out is None, not takes_tuned)
            # tgemm writes bf16, which would round the logits of an fp32 front.
            self.assertIsNone(
                tuned_front_gemm(
                    x=_shaped(64, 7168),
                    weight=front,
                    out_dtype=torch.float32,
                    window=(48, 192),
                )
            )
            # The [router | latent down] pair of a front without shared experts.
            self.assertIsNone(
                tuned_front_gemm(
                    x=_shaped(64, 7168),
                    weight=_shaped(4480, 7168),
                    out_dtype=None,
                    window=(48, 192),
                )
            )

    def test_window_needs_aiter_and_the_switch(self):
        for use_aiter, enabled in ((False, True), (True, False)):
            with (
                self.subTest(use_aiter=use_aiter, enabled=enabled),
                envs.SGLANG_USE_AITER.override(use_aiter),
                envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT.override(enabled),
            ):
                self.assertIsNone(tuned_front_window())
        with (
            envs.SGLANG_USE_AITER.override(True),
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT.override(True),
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MIN_TOKENS.override(8),
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MAX_TOKENS.override(64),
        ):
            self.assertEqual(tuned_front_window(), (8, 64))


if __name__ == "__main__":
    unittest.main()
