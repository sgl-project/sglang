"""Unit tests for ``quark_mla_absorb_to_fp8`` (SGLANG_MLA_ABSORB_FP8).

Validates the per-tensor FP8 (e4m3) requantization of the DeepSeek MLA-absorb
``kv_b_proj`` weight: output dtypes/shapes, the single shared per-tensor scale,
and that dequantizing (``w_fp8.float() * w_scale``) reconstructs the original
BF16 weight within FP8 (e4m3) tolerance.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.quark.utils import quark_mla_absorb_to_fp8
from sglang.test.test_utils import CustomTestCase


class TestQuarkMlaAbsorbToFp8(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.H = 8  # heads
        self.qk_nope = 16
        self.v = 12
        self.L = 32  # kv_lora_rank
        self.self_attn = SimpleNamespace(
            qk_nope_head_dim=self.qk_nope, v_head_dim=self.v
        )

    def _make_bf16_w(self):
        # kv_b_proj weight shape: (H * (qk_nope + v), L)
        return torch.randn(
            self.H * (self.qk_nope + self.v), self.L, dtype=torch.bfloat16
        )

    def test_output_dtypes_shapes_and_scale(self):
        w = self._make_bf16_w()
        w_kc, w_vc, w_scale = quark_mla_absorb_to_fp8(self.self_attn, w)

        self.assertEqual(w_kc.dtype, torch.float8_e4m3fn)
        self.assertEqual(w_vc.dtype, torch.float8_e4m3fn)
        # Split back into per-head (kc, vc) blocks.
        self.assertEqual(tuple(w_kc.shape), (self.H, self.qk_nope, self.L))
        self.assertEqual(tuple(w_vc.shape), (self.H, self.v, self.L))
        # A single shared per-tensor scale of shape (1,).
        self.assertEqual(tuple(w_scale.shape), (1,))
        self.assertTrue(torch.isfinite(w_scale).all())
        self.assertGreater(float(w_scale), 0.0)

    def test_dequant_reconstructs_within_fp8_tolerance(self):
        w = self._make_bf16_w()
        w_kc, w_vc, w_scale = quark_mla_absorb_to_fp8(self.self_attn, w)

        # Reference: the same head-split of the original BF16 weight.
        ref_kc, ref_vc = w.unflatten(0, (-1, self.qk_nope + self.v)).split(
            [self.qk_nope, self.v], dim=1
        )
        deq_kc = w_kc.float() * w_scale
        deq_vc = w_vc.float() * w_scale

        # e4m3 has ~2 bits of mantissa; relative error per element is bounded.
        # Compare on normalized (by amax) tensors with a generous but meaningful
        # tolerance.
        amax = w.abs().float().amax().clamp(min=1e-12)
        self.assertLess((deq_kc - ref_kc.float()).abs().max().item() / amax, 0.10)
        self.assertLess((deq_vc - ref_vc.float()).abs().max().item() / amax, 0.10)

    def test_scale_matches_e4m3_dynamic_range(self):
        # The per-tensor scale should map the weight amax near e4m3 max (448).
        w = self._make_bf16_w()
        _, _, w_scale = quark_mla_absorb_to_fp8(self.self_attn, w)
        amax = w.abs().float().amax().clamp(min=1e-12)
        # w_scale is the dequant multiplier ~= amax / 448.
        self.assertAlmostEqual(
            float(w_scale), float(amax) / 448.0, delta=float(amax) * 0.02
        )


if __name__ == "__main__":
    unittest.main()
