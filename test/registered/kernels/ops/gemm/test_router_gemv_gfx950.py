"""The gfx950 router GEMV keeps fp32 accuracy and is deterministic on both split-K reductions."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

NUM_EXPERTS, HIDDEN = 128, 6144


@unittest.skipUnless(is_gfx95_supported(), "ROCm gfx95 only")
class TestRouterGemv(CustomTestCase):
    def setUp(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        self.w = (
            torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02
        ).to(torch.bfloat16)
        self.gen = gen

    def _x(self, m):
        return torch.randn(m, HIDDEN, device="cuda", generator=self.gen).to(
            torch.bfloat16
        )

    def test_accuracy_and_determinism(self):
        from sglang.kernels.ops.gemm.router_gemv import (
            router_gemv,
            router_gemv_supported,
        )

        for m in (1, 16, 17, 100, 128, 129, 256, 300, 512):
            x = self._x(m)
            self.assertTrue(router_gemv_supported(x, self.w))
            out = router_gemv(x, self.w)
            # bf16 x bf16 products are exact in fp64, so the gap is fp32 summation order
            ref = x.double() @ self.w.double().t()
            torch.testing.assert_close(out.double(), ref, rtol=0, atol=1e-5)
            self.assertTrue(torch.equal(out, router_gemv(x, self.w)))
        self.assertFalse(router_gemv_supported(self._x(513), self.w))

    def test_graph_replay_matches_eager(self):
        """Calls captured back to back share the split-K scratch and fixup counter; a replay
        must still reproduce each call's eager output."""
        from sglang.kernels.ops.gemm.router_gemv import router_gemv

        xs = [self._x(m) for m in (8, 256, 40, 512)]
        eager = [router_gemv(x, self.w) for x in xs]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [router_gemv(x, self.w) for x in xs]
        graph.replay()
        torch.cuda.synchronize()
        for out, ref in zip(outs, eager):
            self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
    unittest.main()
