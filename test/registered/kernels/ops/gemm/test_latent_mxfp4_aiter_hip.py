"""Tests for Kimi-K3 non-EP tuned front and latent MXFP4 adapters."""

import unittest
from unittest.mock import patch

import torch

import sglang.srt.models.kimi_k3 as kimi_k3_model
from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip
from sglang.srt.models.kimi_k3 import KimiK3MoE
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(torch.cuda.is_available(), "no GPU")
class TestKimiK3LatentMXFP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not latent_mxfp4_aiter_hip.supported():
            raise unittest.SkipTest("Kimi-K3 latent MXFP4 needs gfx950")
        torch.manual_seed(0)
        cls.device = torch.device("cuda", 0)
        cls.weight = (
            torch.randn(3584, 7168, dtype=torch.bfloat16, device=cls.device) * 0.02
        )
        cls.packed, cls.scale = latent_mxfp4_aiter_hip.pack(
            cls.weight, "latent down_proj"
        )

    def test_pack_footprint(self):
        expected = latent_mxfp4_aiter_hip.packed_bytes(tuple(self.weight.shape))
        actual = self.packed.numel() * self.packed.element_size()
        actual += self.scale.numel() * self.scale.element_size()
        self.assertEqual(actual, expected)

    def test_run_matches_bf16_reference(self):
        x = torch.randn(8, 7168, dtype=torch.bfloat16, device=self.device) * 0.1
        actual = latent_mxfp4_aiter_hip.run(x, self.packed, self.scale)
        expected = torch.nn.functional.linear(x, self.weight)
        rel_l2 = (
            (actual.float() - expected.float()).norm() / expected.float().norm()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item()
        # Per-1x32 MXFP4 has expected quantization error on random normal
        # inputs; these bounds catch layout/scale corruption while leaving
        # model-level quality to the GSM8K gate.
        self.assertLess(rel_l2, 0.20)
        self.assertGreater(cosine, 0.98)

    def test_graph_replay_uses_changed_inputs(self):
        x = torch.randn(8, 7168, dtype=torch.bfloat16, device=self.device) * 0.1
        latent_mxfp4_aiter_hip.run(x, self.packed, self.scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = latent_mxfp4_aiter_hip.run(x, self.packed, self.scale)
        graph.replay()
        first = out.clone()
        x.copy_(torch.randn_like(x) * 0.1)
        graph.replay()
        second = out.clone()
        self.assertFalse(torch.equal(first, second))

    def test_threshold_is_fail_closed(self):
        moe = KimiK3MoE.__new__(KimiK3MoE)
        moe._front_down_w4 = object()
        moe._front_down_scale4 = object()
        moe._latent_up_w4 = object()
        moe._latent_up_scale4 = object()
        self.assertFalse(moe._use_moe_latent_mxfp4(2047))
        self.assertTrue(moe._use_moe_latent_mxfp4(2048))

    def test_tuned_front_does_not_take_output_buffer_calls(self):
        x = torch.randn(48, 7168, dtype=torch.bfloat16, device=self.device)
        weight = torch.empty(6016, 7168, dtype=torch.bfloat16, device=self.device)
        expected = torch.randn(48, 6016, dtype=torch.bfloat16, device=self.device)
        with (
            patch.object(kimi_k3_model, "_use_aiter", True),
            patch.object(kimi_k3_model, "_k3_aiter_tuned_moe_front", True),
            patch("aiter.tuned_gemm.tgemm.mm", return_value=expected) as tuned_mm,
        ):
            kimi_k3_model._k3_bf16_gemm(x[:47], weight)
            kimi_k3_model._k3_bf16_gemm(
                torch.empty(193, 7168, dtype=torch.bfloat16, device=self.device),
                weight,
            )
            tuned_mm.assert_not_called()
            actual = kimi_k3_model._k3_bf16_gemm(x, weight)
            self.assertIs(actual, expected)
            out = torch.empty_like(expected)
            kimi_k3_model._k3_bf16_gemm(x, weight, out=out)
            tuned_mm.assert_called_once()

    def test_tuned_front_accepts_quark_partial_shape(self):
        x = torch.randn(64, 7168, dtype=torch.bfloat16, device=self.device)
        weight = torch.empty(4480, 7168, dtype=torch.bfloat16, device=self.device)
        expected = torch.randn(64, 4480, dtype=torch.bfloat16, device=self.device)
        with (
            patch.object(kimi_k3_model, "_use_aiter", True),
            patch.object(kimi_k3_model, "_k3_aiter_tuned_moe_front", True),
            patch("aiter.tuned_gemm.tgemm.mm", return_value=expected) as tuned_mm,
        ):
            actual = kimi_k3_model._k3_bf16_gemm(x, weight)
        self.assertIs(actual, expected)
        tuned_mm.assert_called_once_with(x, weight, None, otype=x.dtype)


if __name__ == "__main__":
    unittest.main()
