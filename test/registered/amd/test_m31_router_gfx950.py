"""The fused M3.1 decode router and its aiter-sort hand-off match the unfused path bit for bit."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

E, TOPK, DIM, BLOCK_M = 128, 4, 6144, 16


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(), "gfx950 (MI35x) only"
)
class TestM31Router(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import aiter.fused_moe as fm

        from sglang.kernels.ops.moe import moe_sorting_small
        from sglang.kernels.ops.moe.smallm_router_gfx950 import m31_router

        moe_sorting_small.apply_aiter_small_moe_sort_patch()
        cls.fm = fm
        cls.S = moe_sorting_small
        cls.R = m31_router
        cls.dev = torch.device("cuda", 0)
        g = torch.Generator(device="cpu").manual_seed(0)
        cls.gate = (
            (torch.randn(E, DIM, generator=g) * 0.02).to(torch.bfloat16).to(cls.dev)
        )
        cls.bias = (torch.randn(E, generator=g) * 0.05).to(cls.dev)

    def _x(self, m, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(m, DIM, generator=g).to(torch.bfloat16).to(self.dev)

    def _chain(self, x):
        from sglang.kernels.ops.gemm.router_gemv import router_gemv
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

        logits = router_gemv(x, self.gate)
        w, ids = moe_fused_gate(
            logits,
            self.bias,
            topk=TOPK,
            scoring_func="sigmoid",
            num_fused_shared_experts=0,
            renormalize=True,
            routed_scaling_factor=2.0,
            apply_routed_scaling_factor_on_output=True,
        )
        sort = self.fm.moe_sorting(ids, w, E, DIM, torch.bfloat16, block_size=BLOCK_M)
        return logits, w, ids, sort

    def _assert_same(self, a, b):
        if a.dtype == torch.float32:
            a, b = a.view(torch.int32), b.view(torch.int32)
        self.assertTrue(torch.equal(a, b))

    def _assert_matches_chain(self, fused, chain):
        f_w, f_ids, f_logits, f_sort = fused
        c_logits, c_w, c_ids, c_sort = chain
        self._assert_same(f_logits, c_logits)
        self._assert_same(f_ids, c_ids)
        self._assert_same(f_w, c_w)
        self._assert_same(f_sort[3], c_sort[3])
        nv = int(c_sort[3][0])
        self._assert_same(f_sort[0][:nv], c_sort[0][:nv])
        self._assert_same(f_sort[1][:nv], c_sort[1][:nv])
        self._assert_same(f_sort[2][: nv // BLOCK_M], c_sort[2][: nv // BLOCK_M])

    def test_bitwise_vs_chain(self):
        for m in range(1, 17):
            for seed in range(3):
                x = self._x(m, 100 * m + seed)
                self.assertTrue(self.R.m31_router_supported(x, self.gate, self.bias))
                fused = self.R.m31_router(x, self.gate, self.bias, 2.0)
                self._assert_matches_chain(fused, self._chain(x))

    def test_presorted_handoff(self):
        """The sort hands over the router's buffers; its stage2 output is zero-filled only
        for a stage2 accumulating into it, and a pending addend stands in for it."""
        from sglang.kernels.ops.moe.aiter_moe_addend import moe_output_addend

        x = self._x(9, 7)
        w, ids, _, sort = self.R.m31_router(x, self.gate, self.bias, 2.0)
        sort[4].fill_(7.0)
        with self.S.presorted_moe_sort(ids, sort, num_experts=E, block_size=BLOCK_M):
            got = self.fm.moe_sorting(
                ids, w, E, DIM, torch.bfloat16, block_size=BLOCK_M
            )
            # another topk_ids tensor is sorted, not handed the buffers
            other = self.fm.moe_sorting(
                ids.clone(), w, E, DIM, torch.bfloat16, block_size=BLOCK_M
            )
        for a, b in zip(got, sort):
            self.assertIs(a, b)
        self.assertEqual(int(got[4].count_nonzero()), 0)
        self.assertIsNot(other[0], sort[0])

        addend = torch.full_like(sort[4], 3.0)
        with (
            self.S.presorted_moe_sort(ids, sort, num_experts=E, block_size=BLOCK_M),
            moe_output_addend(addend) as slot,
        ):
            got = self.fm.moe_sorting(
                ids, w, E, DIM, torch.bfloat16, block_size=BLOCK_M
            )
        self.assertTrue(slot.consumed)
        self.assertIs(got[4], addend)
        self.assertTrue(bool((addend == 3.0).all()))

    def _shared_gate_up(self, m, seed):
        """per-token fp8 activations and a shuffled [1536, 6144] fp8 weight with scales"""
        from aiter.ops.shuffle import shuffle_weight

        g = torch.Generator(device="cpu").manual_seed(seed)
        xq = (
            (torch.randn(m, DIM, generator=g) * 64).to(torch.float8_e4m3fn).to(self.dev)
        )
        xs = (torch.rand(m, 1, generator=g) * 0.01 + 1e-3).to(self.dev)
        w = (torch.randn(self.R.SHARED_N, DIM, generator=g) * 64).to(
            torch.float8_e4m3fn
        )
        ws = (torch.rand(self.R.SHARED_N, 1, generator=g) * 0.01 + 1e-3).to(self.dev)
        return xq, xs, shuffle_weight(w.to(self.dev), (16, 16)), ws

    def _assert_shared_matches(self, fused, x, xq, xs, wq, ws):
        from sglang.kernels.ops.gemm.smallm_fp8_gfx950 import smallm_fp8_gemm

        gate_up, routed = fused
        self._assert_same(
            gate_up.view(torch.int16),
            smallm_fp8_gemm(xq, wq, xs, ws, torch.bfloat16).view(torch.int16),
        )
        self._assert_matches_chain(routed, self._chain(x))

    def test_shared_router_bitwise(self):
        """The router + shared expert gate_up launch equals smallm_fp8_gemm and the router chain."""
        for m in range(3, 17):
            for seed in range(2):
                x = self._x(m, 100 * m + seed)
                xq, xs, wq, ws = self._shared_gate_up(m, 7 * m + seed)
                self.assertTrue(
                    self.R.m31_shared_router_supported(
                        x, self.gate, self.bias, xq, xs, wq
                    )
                )
                fused = self.R.m31_shared_router(
                    x, self.gate, self.bias, 2.0, xq, xs, wq, ws
                )
                self._assert_shared_matches(fused, x, xq, xs, wq, ws)

    def test_shared_router_cuda_graph_replay(self):
        for m in (8, 16):
            x = self._x(m, 1)
            xq, xs, wq, ws = self._shared_gate_up(m, 2)
            # workspace and build outside capture
            self.R.m31_shared_router(x, self.gate, self.bias, 2.0, xq, xs, wq, ws)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.R.m31_shared_router(
                    x, self.gate, self.bias, 2.0, xq, xs, wq, ws
                )
            for seed in range(3):
                x.copy_(self._x(m, 50 + seed))
                xq.copy_(self._shared_gate_up(m, 60 + seed)[0])
                g.replay()
                torch.cuda.synchronize()
                self._assert_shared_matches(out, x, xq, xs, wq, ws)

    def test_cuda_graph_replay(self):
        x = self._x(12, 1)
        # workspace and build outside capture
        self.R.m31_router(x, self.gate, self.bias, 2.0)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.R.m31_router(x, self.gate, self.bias, 2.0)
        for seed in range(3):
            x.copy_(self._x(12, 50 + seed))
            g.replay()
            torch.cuda.synchronize()
            self._assert_matches_chain(out, self._chain(x))

    def test_build_is_cached_across_loads(self):
        """One hipcc build per source serves every later load (restarts, other ranks) and
        leaves nothing else in the cache."""
        import subprocess
        import tempfile
        from unittest.mock import patch

        from sglang.srt.environ import envs

        builds = []
        real_run = subprocess.run

        def run(cmd, **kwargs):
            builds.append(cmd)
            return real_run(cmd, **kwargs)

        with (
            tempfile.TemporaryDirectory() as cache,
            envs.SGLANG_JIT_CACHE_DIR.override(cache),
            patch.object(self.R.subprocess, "run", run),
        ):
            first = self.R._code_object()
            self.assertEqual(self.R._code_object(), first)
            self.assertEqual(len(builds), 1)
            self.assertEqual(
                sorted(p.name for p in first.parent.iterdir()), [first.name]
            )


if __name__ == "__main__":
    unittest.main()
