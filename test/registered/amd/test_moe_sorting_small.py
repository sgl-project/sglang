"""Small-batch MoE sorting patch (sglang.kernels.ops.moe.moe_sorting_small) vs stock aiter sorting.

Runs only on ROCm gfx950 with aiter; everywhere else the test is skipped.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _gfx950():
    return (
        bool(torch.version.hip)
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    )


@unittest.skipUnless(_gfx950(), "gfx950 (MI35x) only")
class TestMoeSortingSmall(CustomTestCase):
    # Qwen3.5: 512 routed experts + fused shared expert, top-10 + shared slot
    E, TOPK, DIM = 513, 11, 4096

    @classmethod
    def setUpClass(cls):
        import aiter.fused_moe as fm

        from sglang.kernels.ops.moe import moe_sorting_small as S

        cls.S = S
        cls.fm = fm
        orig_sort = fm._moe_sorting_impl
        orig_quant = fm.fused_dynamic_mxfp8_quant_moe_sort
        S.apply_aiter_small_moe_sort_patch()
        if fm._moe_sorting_impl is orig_sort:
            # an earlier import already patched aiter; recover the originals
            orig_sort = fm._moe_sorting_impl.__wrapped__
            orig_quant = fm.fused_dynamic_mxfp8_quant_moe_sort.__wrapped__
        cls.orig_sort = staticmethod(orig_sort)
        cls.orig_quant = staticmethod(orig_quant)
        cls.dev = torch.device("cuda", 0)

    def _routing(self, m, seed, E=E, topk=TOPK):
        g = torch.Generator(device="cpu").manual_seed(seed)
        ids = torch.stack(
            [torch.randperm(E - 1, generator=g)[: topk - 1] for _ in range(m)]
        )
        ids = torch.cat([ids, torch.full((m, 1), E - 1)], 1)
        ids = ids.to(torch.int32).to(self.dev).contiguous()
        return ids, torch.rand(m, topk, generator=g).to(self.dev)

    def _sort(self, fn, ids, w, bs, E=E):
        return fn(ids, w, E, self.DIM, torch.bfloat16, bs, None, None, 0, True)

    def _emitted_quant(self, ids, w, x, bs, E):
        S = self.S
        tok = S._pending_quant_input.set(x)
        etok = S._emitted_quant.set(None)
        try:
            self._sort(self.fm._moe_sorting_impl, ids, w, bs, E)
            return S._emitted_quant.get()
        finally:
            S._emitted_quant.reset(etok)
            S._pending_quant_input.reset(tok)

    def _assert_same_sort(self, a, b, bs, msg):
        na, nb = int(a[3][0]), int(b[3][0])
        self.assertTrue(torch.equal(a[3].cpu(), b[3].cpu()), msg)
        self.assertTrue(
            torch.equal(a[2][: na // bs].cpu(), b[2][: nb // bs].cpu()), msg
        )
        ia, wa = a[0][:na].tolist(), a[1][:na].tolist()
        ib, wb = b[0][:nb].tolist(), b[1][:nb].tolist()
        # the order of pairs inside one expert block is not part of the contract
        for i in range(0, na, bs):
            self.assertEqual(
                sorted(zip(ia[i : i + bs], wa[i : i + bs])),
                sorted(zip(ib[i : i + bs], wb[i : i + bs])),
                f"{msg} block {i // bs}",
            )

    def test_sort_matches_aiter(self):
        # compact at E=513; distributed at MiniMax-M3's E=129 / top-4 + shared
        cases = [(513, 11, m) for m in (1, 2, 4, 5)] + [
            (129, 5, m) for m in (13, 25, 51)
        ]
        for bs in (32, 64):
            for E, topk, m in cases:
                ids, w = self._routing(m, m, E, topk)
                self._assert_same_sort(
                    self._sort(self.orig_sort, ids, w, bs, E),
                    self._sort(self.fm._moe_sorting_impl, ids, w, bs, E),
                    bs,
                    f"E={E} m={m} bs={bs}",
                )

    def test_fused_mxfp8_quant_matches_aiter(self):
        bs = 32
        # compact at E=513; distributed at MiniMax-M3's E=129 / top-4 + shared
        for E, topk, m in ((513, 11, 1), (513, 11, 5), (129, 5, 16), (129, 5, 51)):
            ids, w = self._routing(m, m, E, topk)
            x = torch.randn(m, self.DIM, dtype=torch.bfloat16, device=self.dev)
            sid, sw, _, nv, _ = self._sort(self.orig_sort, ids, w, bs, E)
            ref_q, _ = self.orig_quant(
                x,
                sorted_ids=sid,
                num_valid_ids=nv,
                token_num=m,
                topk=topk,
                block_size=bs,
                sorted_weights=sw,
                num_experts_upper_bound=E,
            )
            emitted = self._emitted_quant(ids, w, x, bs, E)
            self.assertIsNotNone(emitted, f"E={E} m={m}")
            self.assertTrue(
                torch.equal(ref_q.view(torch.uint8), emitted[0].view(torch.uint8)),
                f"E={E} m={m}",
            )

    def test_falls_back_to_aiter_past_compact_at_513_experts(self):
        m = 6
        ids, w = self._routing(m, m)
        self.assertFalse(self.S._small_sort_supported(ids, 32, None, None, self.E))
        x = torch.randn(m, self.DIM, dtype=torch.bfloat16, device=self.dev)
        self.assertIsNone(self._emitted_quant(ids, w, x, 32, self.E))


if __name__ == "__main__":
    unittest.main()
