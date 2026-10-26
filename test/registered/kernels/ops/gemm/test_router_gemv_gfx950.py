"""The gfx950 router GEMV keeps fp32 accuracy and is deterministic on both split-K reductions."""

import unittest
from functools import partial

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

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

    def test_gate_sums_partials_like_the_reduce_launch(self):
        """moe_fused_gate on router_gemv's split-K partials routes as on its logits."""
        from sglang.kernels.ops.gemm.router_gemv import (
            router_gemv,
            router_gemv_partials,
        )
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

        bias = torch.randn(NUM_EXPERTS, device="cuda", generator=self.gen) * 0.05
        gate = partial(
            moe_fused_gate,
            bias=bias,
            topk=4,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=2.0,
            apply_routed_scaling_factor_on_output=True,
        )
        self.assertIsNone(router_gemv_partials(self._x(16), self.w))
        for m in (17, 128, 300, 512):
            x = self._x(m)
            ref = gate(router_gemv(x, self.w))
            out = gate(router_gemv_partials(x, self.w))
            for got, want in zip(out, ref):
                self.assertTrue(torch.equal(got, want))

    def test_partials_with_swiglu_bitwise(self):
        """The split-K partials and the shared expert's fp8 SwiGLU in one launch equal
        router_gemv_partials and swiglu_oai_fp8, eagerly and in graph replay."""
        from sglang.kernels.ops.gemm.router_gemv import (
            router_gemv_partials,
            router_gemv_partials_supported,
            router_gemv_partials_swiglu,
        )
        from sglang.kernels.ops.quantization.aiter_fp8_quant_triton import (
            swiglu_oai_fp8,
        )

        def gate_up(m, amp):
            return (torch.randn(m, 1536, device="cuda", generator=self.gen) * amp).to(
                torch.bfloat16
            )

        def check(fused, x, gu, quant):
            ref_q, ref_s = swiglu_oai_fp8(gu, 1.702, 7.0, *quant)
            ref_p = router_gemv_partials(x, self.w)
            partials, (q, s) = fused
            self.assertTrue(torch.equal(partials, ref_p))
            self.assertTrue(torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8)))
            self.assertTrue(torch.equal(s, ref_s))
            self.assertEqual(s.stride(), ref_s.stride())

        self.assertFalse(router_gemv_partials_supported(self._x(16), self.w))
        # every router_gemv bucket past the fixup, per-token and 1 x 128 quant
        for m in (17, 24, 64, 96, 128, 129, 256, 300, 320, 512):
            for quant in ((None, False), (128, True)):
                for amp in (1.0, 30.0):
                    x, gu = self._x(m), gate_up(m, amp)
                    self.assertTrue(router_gemv_partials_supported(x, self.w))
                    fused = router_gemv_partials_swiglu(
                        x, self.w, gu, 1.702, 7.0, *quant
                    )
                    fused = (fused[0].clone(), fused[1])
                    check(fused, x, gu, quant)

        x, gu = self._x(64), gate_up(64, 4.0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused = router_gemv_partials_swiglu(x, self.w, gu, 1.702, 7.0)
        for _ in range(2):
            x.copy_(self._x(64))
            gu.copy_(gate_up(64, 4.0))
            graph.replay()
            torch.cuda.synchronize()
            check((fused[0].clone(), fused[1]), x, gu, (None, False))

    def test_graph_replay_matches_eager(self):
        """Back-to-back calls share the split-K scratch."""
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
