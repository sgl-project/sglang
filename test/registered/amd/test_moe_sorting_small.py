"""Small-batch MoE sorting patch (sglang.kernels.ops.moe.moe_sorting_small) vs stock aiter sorting."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=80, suite="stage-b-test-1-gpu-small-amd-mi35x")


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
        ids, _ = self._routing(m, m)
        self.assertFalse(self.S._small_sort_supported(ids, 32, None, None, self.E))


def _reference(ids, w, num_experts, block_size):
    m, topk = ids.shape
    flat = ids.flatten().tolist()
    sorted_ids, sorted_w, expert_ids = [], [], []
    for e in range(num_experts):
        pairs = [p for p, x in enumerate(flat) if x == e]
        if not pairs:
            continue
        n = -(-len(pairs) // block_size) * block_size
        sorted_ids += [((p % topk) << 24) | (p // topk) for p in pairs]
        sorted_ids += [(topk << 24) | m] * (n - len(pairs))
        sorted_w += [w.flatten()[p].item() for p in pairs] + [0.0] * (n - len(pairs))
        expert_ids += [e] * (n // block_size)
    return sorted_ids, sorted_w, expert_ids


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "ROCm only")
class TestMoeSortingSmallNoQuant(CustomTestCase):
    """No-quant path of _run_small_sort (incl. the histogram + bitonic kernel for
    32 < P <= 256) vs a torch reference of the aiter moe_sorting layout."""

    def _check(self, m, topk, num_experts, block_size, hot):
        from sglang.kernels.ops.moe.moe_sorting_small import _run_small_sort

        g = torch.Generator().manual_seed(m * 1000 + topk)
        ids = torch.stack(
            [torch.randperm(hot, generator=g)[:topk] for _ in range(m)]
        ).to(torch.int32)
        ids = (ids + num_experts - hot).cuda()  # hot = the top `hot` expert ids
        w = torch.rand(m, topk, generator=g).cuda()
        max_padded = m * topk + num_experts * block_size - topk
        sorted_ids = torch.full((max_padded,), -7, dtype=torch.int32, device="cuda")
        sorted_w = torch.full((max_padded,), -7.0, device="cuda")
        expert_ids = torch.full(
            (-(-max_padded // block_size),), -7, dtype=torch.int32, device="cuda"
        )
        num_valid = torch.full((2,), -7, dtype=torch.int32, device="cuda")
        moe_buf = torch.ones(m, 7168, dtype=torch.bfloat16, device="cuda")
        args = (sorted_ids, sorted_w, expert_ids, num_valid, moe_buf, block_size)
        _run_small_sort(ids, w, *args, None, num_experts)
        ref_ids, ref_w, ref_e = _reference(ids.cpu(), w.cpu(), num_experts, block_size)
        n = len(ref_ids)
        self.assertEqual(num_valid.tolist(), [n, m])
        self.assertEqual(sorted_ids[:n].tolist(), ref_ids)
        self.assertEqual(sorted_w[:n].tolist(), ref_w)
        self.assertEqual(expert_ids[: n // block_size].tolist(), ref_e)
        self.assertTrue(bool((moe_buf == 0).all()))

    def test_no_quant(self):
        for num_experts, topk, block_size in ((385, 7, 32), (257, 9, 32), (129, 8, 16)):
            for m in range(1, 256 // topk + 1):
                for hot in (num_experts, 12):  # uniform and skewed routing
                    with self.subTest(e=num_experts, topk=topk, m=m, hot=hot):
                        self._check(m, topk, num_experts, block_size, hot)


@unittest.skipUnless(_gfx950(), "gfx950 (MI35x) only")
class TestMoeSortQuant(CustomTestCase):
    """The one-launch sort (+ MXFP4 stage1 quant) must equal aiter's two-phase sort and
    fused_dynamic_mxfp4_quant_moe_sort byte for byte (MiniMax-M3.1 TP4: 128 experts, top-4)."""

    E, TOPK, DIM = 128, 4, 6144

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

    def _routing(self, m, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        ids = torch.stack(
            [torch.randperm(self.E, generator=g)[: self.TOPK] for _ in range(m)]
        )
        w = torch.rand(m, self.TOPK, generator=g) + 0.05
        return ids.to(torch.int32).to(self.dev), w.to(self.dev)

    def _aiter_sort(self, ids, w, bs):
        return self.orig_sort(
            ids,
            w,
            self.E,
            self.DIM,
            torch.bfloat16,
            bs,
            None,
            None,
            2,
            True,
            accumulate=False,
        )

    def test_sort_matches_aiter(self):
        from sglang.kernels.ops.moe.moe_sort_quant import moe_sort_quant

        empty = torch.empty((0, 0), dtype=torch.bfloat16, device=self.dev)
        for bs in (16, 32, 64):
            for m in (17, 64, 200, 384, 1000):
                ids, w = self._routing(m, m + bs)
                ref = self._aiter_sort(ids, w, bs)
                out = moe_sort_quant(ids, w, self.E, bs, empty)
                nv = int(ref[3][0])
                msg = f"m={m} bs={bs}"
                self.assertTrue(torch.equal(ref[3], out[3]), msg)
                self.assertTrue(torch.equal(ref[0][:nv], out[0][:nv]), msg)
                self.assertTrue(torch.equal(ref[1][:nv], out[1][:nv]), msg)
                self.assertTrue(
                    torch.equal(ref[2][: nv // bs], out[2][: nv // bs]), msg
                )

    def test_fp4_quant_matches_aiter(self):
        from sglang.kernels.ops.moe.moe_sort_quant import moe_sort_quant

        empty = torch.empty((0, 0), dtype=torch.bfloat16, device=self.dev)
        for m, bs in ((256, 32), (384, 32), (1000, 64)):
            ids, w = self._routing(m, m)
            x = torch.randn(m, self.DIM, dtype=torch.bfloat16, device=self.dev)
            sid, sw, _, nv, _ = self._aiter_sort(ids, w, bs)
            ref_q, ref_s = self.orig_fp4_quant(
                x,
                sorted_ids=sid,
                num_valid_ids=nv,
                token_num=m,
                topk=self.TOPK,
                block_size=bs,
                sorted_weights=sw,
                num_experts_upper_bound=self.E,
            )
            out = moe_sort_quant(ids, w, self.E, bs, empty, x=x)
            msg = f"m={m} bs={bs}"
            self.assertTrue(
                torch.equal(ref_q.view(torch.uint8), out[4].view(torch.uint8)), msg
            )
            # aiter leaves the scale rows of padding slots unwritten; compare the real ones
            n = int(nv[0])
            rows = torch.arange(n, device=self.dev)[(sid[:n] & 0xFFFFFF) < m][:, None]
            y = torch.arange(self.DIM // 32, device=self.dev)[None, :]
            addr = (
                (rows // 32) * (192 * 32)
                + (y // 8) * 256
                + (y % 4) * 64
                + (rows % 16) * 4
                + ((y % 8) // 4) * 2
                + (rows % 32) // 16
            )
            self.assertTrue(
                torch.equal(
                    ref_s.view(torch.uint8).flatten()[addr],
                    out[5].view(torch.uint8).flatten()[addr],
                ),
                msg,
            )

    def test_patched_sort_hands_its_quant_to_aiter(self):
        """The patched sort emits the stage1 quant, which the patched quant call returns."""
        m, bs = 384, 32
        ids, w = self._routing(m, 1)
        x = torch.randn(m, self.DIM, dtype=torch.bfloat16, device=self.dev)
        S = self.S
        tok = S._pending_fp4_input.set(x)
        etok = S._emitted_fp4_quant.set(None)
        try:
            sid, sw, _, nv, _ = self.fm._moe_sorting_impl(
                ids,
                w,
                self.E,
                self.DIM,
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
                topk=self.TOPK,
                block_size=bs,
                sorted_weights=sw,
                num_experts_upper_bound=self.E,
            )
            self.assertIs(q, emitted[0])
            self.assertIs(scale, emitted[1])
        finally:
            S._emitted_fp4_quant.reset(etok)
            S._pending_fp4_input.reset(tok)


if __name__ == "__main__":
    unittest.main()
