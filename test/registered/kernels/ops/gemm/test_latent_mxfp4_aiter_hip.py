"""Tests for Kimi-K3 non-EP tuned front and latent MXFP4 adapters."""

import unittest

import torch

from sglang.kernels.ops.gemm import latent_mxfp4_aiter_hip
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
        cls.packed, cls.scale = latent_mxfp4_aiter_hip.pack(cls.weight)

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


if __name__ == "__main__":
    unittest.main()
