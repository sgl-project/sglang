"""Prefill-sized MoE sort + MXFP4 stage1 quant (sglang.kernels.ops.moe.moe_sort_quant_large)
vs stock aiter sorting and its split MXFP4 quant."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(), "gfx950 (MI35x) only"
)
class TestMoeSortQuantLarge(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import aiter.fused_moe as fm

        from sglang.kernels.ops.moe import moe_sorting_small as S

        S.apply_aiter_small_moe_sort_patch()
        cls.S = S
        cls.fm = fm
        cls.orig_sort = staticmethod(fm._moe_sorting_impl.__wrapped__)
        cls.orig_fp4_quant = staticmethod(
            fm.fused_dynamic_mxfp4_quant_moe_sort.__wrapped__
        )
        cls.dev = torch.device("cuda", 0)

    def _inputs(self, m, num_experts, topk, dim, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        ids = torch.stack(
            [torch.randperm(num_experts, generator=g)[:topk] for _ in range(m)]
        )
        w = torch.rand(m, topk, generator=g) + 0.05
        x = torch.randn(m, dim, generator=g).to(torch.bfloat16)
        return (
            ids.to(torch.int32).to(self.dev),
            w.to(self.dev),
            x.to(self.dev),
        )

    def _aiter(self, ids, w, x, num_experts, bs):
        sid, sw, seid, nv, _ = self.orig_sort(
            ids,
            w,
            num_experts,
            x.shape[1],
            torch.bfloat16,
            bs,
            None,
            None,
            2,
            True,
            accumulate=False,
        )
        q, s = self.orig_fp4_quant(
            x,
            sorted_ids=sid,
            num_valid_ids=nv,
            token_num=x.shape[0],
            topk=ids.shape[1],
            block_size=bs,
        )
        return sid, sw, seid, nv, q, s

    def _scale_addr(self, n, dim):
        rows = torch.arange(n, device=self.dev)[:, None]
        y = torch.arange(dim // 32, device=self.dev)[None, :]
        return (
            (rows // 32) * (dim // 32 * 32)
            + (y // 8) * 256
            + (y % 4) * 64
            + (rows % 16) * 4
            + ((y % 8) // 4) * 2
            + (rows % 32) // 16
        )

    def test_matches_aiter(self):
        from sglang.kernels.ops.moe.moe_sort_quant_large import moe_sort_quant_large

        cases = (
            (1025, 128, 4, 6144, 64),
            (4096, 128, 4, 6144, 128),
        )
        for m, num_experts, topk, dim, bs in cases:
            msg = f"m={m} E={num_experts} topk={topk} dim={dim} bs={bs}"
            ids, w, x = self._inputs(m, num_experts, topk, dim, m + bs)
            ref = self._aiter(ids, w, x, num_experts, bs)
            moe_buf = torch.ones(m, 128, dtype=torch.bfloat16, device=self.dev)
            out = moe_sort_quant_large(ids, w, num_experts, bs, moe_buf, x)
            n = int(ref[3][0])
            self.assertTrue(torch.equal(ref[3], out[3]), msg)
            self.assertTrue(torch.equal(ref[0][:n], out[0][:n]), msg)
            self.assertTrue(torch.equal(ref[1][:n], out[1][:n]), msg)
            self.assertTrue(torch.equal(ref[2][: n // bs], out[2][: n // bs]), msg)
            self.assertTrue(
                torch.equal(ref[4].view(torch.uint8), out[4].view(torch.uint8)), msg
            )
            # aiter's scatter-path scale sort leaves padding rows unwritten; this writes 0
            addr = self._scale_addr(n, dim)
            real = ((ref[0][:n] & 0xFFFFFF) < m)[:, None].expand_as(addr)
            ref_s = ref[5].view(torch.uint8).flatten()
            out_s = out[5].view(torch.uint8).flatten()
            self.assertTrue(torch.equal(ref_s[addr[real]], out_s[addr[real]]), msg)
            self.assertTrue(bool((out_s[addr[~real]] == 0).all()), msg)
            self.assertTrue(bool((moe_buf == 0).all()), msg)

    def test_patched_sort_hands_its_quant_to_aiter(self):
        """Past the one-launch sort's cap, the patched sort emits the stage1 quant that the
        patched quant call returns."""
        m, num_experts, topk, dim, bs = 2048, 128, 4, 6144, 64
        ids, w, x = self._inputs(m, num_experts, topk, dim, 7)
        S = self.S
        tok = S._pending_fp4_input.set(x)
        etok = S._emitted_fp4_quant.set(None)
        try:
            sid, sw, _, nv, _ = self.fm._moe_sorting_impl(
                ids,
                w,
                num_experts,
                dim,
                torch.bfloat16,
                bs,
                None,
                None,
                2,
                True,
                accumulate=False,
            )
            emitted = S._emitted_fp4_quant.get()
            self.assertIsNotNone(emitted)
            q, scale = self.fm.fused_dynamic_mxfp4_quant_moe_sort(
                x,
                sorted_ids=sid,
                num_valid_ids=nv,
                token_num=m,
                topk=topk,
                block_size=bs,
                sorted_weights=sw,
                num_experts_upper_bound=num_experts,
            )
            self.assertIs(q, emitted[0])
            self.assertIs(scale, emitted[1])
        finally:
            S._emitted_fp4_quant.reset(etok)
            S._pending_fp4_input.reset(tok)


if __name__ == "__main__":
    unittest.main()
