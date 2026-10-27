"""Small-batch MoE sorting patch (sglang.kernels.ops.moe.moe_sorting_small) vs stock aiter sorting."""

import os
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=150, suite="stage-b-test-1-gpu-small-amd-mi35x")


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
        # compact kernel up to 32 pairs, sorted kernel above
        cases = (
            [(513, 11, m) for m in (1, 2, 4, 5)]
            + [(129, 5, m) for m in (13, 25, 51)]
            + [(385, 7, m) for m in (4, 5, 10, 36)]
            # past 256 pairs, up to the 1024-pair limit
            + [(129, 5, m) for m in (52, 103, 204)]
            + [(385, 7, m) for m in (37, 74, 146)]
        )
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
        # compact at E=513; sorted kernel above 32 pairs at E=129 and E=385;
        # hidden 5120 (DeepSeek-V4.1) quantizes in 1024-column chunks
        for E, topk, dim, m in (
            (513, 11, 4096, 1),
            (513, 11, 4096, 5),
            (129, 5, 4096, 16),
            (129, 5, 4096, 51),
            (385, 7, 5120, 9),
            (385, 7, 5120, 10),
            (385, 7, 5120, 36),
            (129, 5, 5120, 16),
            (385, 7, 5120, 37),
            (385, 7, 5120, 146),
            (129, 5, 4096, 204),
        ):
            ids, w = self._routing(m, m, E, topk)
            x = torch.randn(m, dim, dtype=torch.bfloat16, device=self.dev)
            sid, sw, _, nv, _ = self._sort(self.orig_sort, ids, w, bs, E)
            ref_q, ref_s = self.orig_quant(
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
            # e8m0 byte of every (real sorted row, 32-group) at aiter's swizzled address
            rows = torch.nonzero(sid[: int(nv[0])] != ((topk << 24) | m))
            pad, g = (dim // 32 + 7) // 8 * 8, torch.arange(dim // 32, device=self.dev)
            addr = (rows // 32) * pad * 32 + rows % 16 * 4 + rows % 32 // 16
            addr = (addr + g // 8 * 256 + g % 4 * 64 + g % 8 // 4 * 2).flatten()
            self.assertTrue(
                torch.equal(
                    ref_s.view(torch.uint8).flatten()[addr],
                    emitted[1].view(torch.uint8).flatten()[addr],
                ),
                f"E={E} m={m} scales",
            )

    def test_falls_back_to_aiter_past_compact_at_513_experts(self):
        m = 6
        ids, _ = self._routing(m, m)
        self.assertFalse(self.S._small_sort_supported(ids, 32, None, None, self.E))

    def test_sorted_path_past_compact_at_385_experts(self):
        # 146 rows = 1022 pairs; 147 rows = 1029 pairs, past the limit
        ids, _ = self._routing(146, 146, 385, 7)
        self.assertTrue(self.S._small_sort_supported(ids, 32, None, None, 385))
        ids, _ = self._routing(147, 147, 385, 7)
        self.assertFalse(self.S._small_sort_supported(ids, 32, None, None, 385))

    def _mxfp4_experts(self, E, hidden=5120, inter=640):
        """Random MXFP4 experts; the default is DeepSeek-V4.1's TP4 shape, hidden 5120 and
        inter 576 padded to 640."""
        from aiter import dtypes
        from aiter.ops.shuffle import shuffle_scale

        g = torch.Generator(device=self.dev).manual_seed(E)

        def rand(*shape, low=0, high=256):
            return torch.randint(
                low, high, shape, dtype=torch.uint8, device=self.dev, generator=g
            )

        w1 = rand(E, 2 * inter, hidden // 2).view(dtypes.fp4x2)
        w2 = rand(E, hidden, inter // 2).view(dtypes.fp4x2)
        w1.is_shuffled = w2.is_shuffled = True
        # e8m0 2^-9 .. 2^-4, shuffled as the checkpoint loader does
        s1 = rand(E * 2 * inter, hidden // 32, low=118, high=124).view(dtypes.fp8_e8m0)
        s2 = rand(E * hidden, inter // 32, low=118, high=124).view(dtypes.fp8_e8m0)
        return (
            w1,
            w2,
            shuffle_scale(s1, E, True, True),
            shuffle_scale(s2, E, True, False),
        )

    def _stage1_out(
        self, x, ids, w, experts, stock, expert_mask=None, pad=64, swiglu_limit=10.0
    ):
        """fused_moe's stage-1 GEMM output (the stage-2 quant input), and whether the
        stage-1 quant came from the sort. The defaults are DeepSeek-V4.1's."""
        from aiter import ActivationType, QuantType
        from aiter.ops.flydsl.moe_common import GateMode

        fm, S = self.fm, self.S
        quant = self.orig_quant if stock else fm.fused_dynamic_mxfp8_quant_moe_sort
        calls = []

        def recording_quant(input, *args, **kwargs):
            calls.append((input.clone(), S._emitted_quant.get() is not None))
            return quant(input, *args, **kwargs)

        aiter_fns = {"fused_dynamic_mxfp8_quant_moe_sort": recording_quant}
        if stock:
            aiter_fns["_moe_sorting_impl"] = self.orig_sort
        kwargs = {} if expert_mask is None else {"expert_mask": expert_mask}
        if swiglu_limit is not None:
            kwargs["swiglu_limit"] = swiglu_limit
        # fp8 activations at every row count, as the DeepSeek-V4.1 recipe sets
        with (
            mock.patch.multiple(fm, **aiter_fns),
            mock.patch.dict(os.environ, {"AITER_BF16_FP8_MOE_BOUND": "0"}),
        ):
            (fm.fused_moe.__wrapped__ if stock else fm.fused_moe)(
                x,
                experts[0],
                experts[1],
                w,
                ids,
                quant_type=QuantType.per_1x32,
                activation=ActivationType.Silu,
                w1_scale=experts[2],
                w2_scale=experts[3],
                intermediate_pad=pad,
                gate_mode=GateMode.INTERLEAVE.value,
                **kwargs,
            )
        # quant of x for stage 1, then of the stage-1 output for stage 2
        self.assertEqual(len(calls), 2)
        return calls[1][0], calls[0][1]

    def test_fused_moe_a8w4_stage1_matches_aiter(self):
        # DeepSeek-V4.1 TP4 decode MoE and its DSpark draft MoE (E=129 / top-3 + shared):
        # compact sort at 1 row, sorted kernel above 32 pairs, up to the 1024-pair limit
        for E, topk, rows in ((385, 7, (1, 16, 36, 37, 146)), (129, 4, (1, 16, 256))):
            experts = self._mxfp4_experts(E)
            for m in rows:
                ids, w = self._routing(m, m, E, topk)
                x = torch.randn(m, 5120, dtype=torch.bfloat16, device=self.dev)
                ref, _ = self._stage1_out(x, ids, w, experts, stock=True)
                out, fused = self._stage1_out(x, ids, w, experts, stock=False)
                self.assertTrue(fused, f"E={E} m={m}: the sort did not emit the quant")
                self.assertTrue(torch.equal(out, ref), f"E={E} m={m}")


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


@unittest.skipUnless(_gfx950(), "gfx950 (MI35x) only")
class TestMoeSortingSmallExpertMask(CustomTestCase):
    # MiMo-V2.6-Pro: 384 routed experts, top-8, hidden 6144, EP8 = 48 experts per rank
    EP_E, EP_TOPK, EP_DIM, EP_SIZE = 384, 8, 6144, 8

    setUpClass = TestMoeSortingSmall.__dict__["setUpClass"]
    _routing = TestMoeSortingSmall._routing
    _assert_same_sort = TestMoeSortingSmall._assert_same_sort

    def _ep_mask(self, E, rank, ep_size):
        mask = torch.zeros(E, dtype=torch.int32)
        n = E // ep_size
        mask[rank * n : (rank + 1) * n] = 1
        return mask.to(self.dev)

    def _random_mask(self, E, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return (torch.rand(E, generator=g) < 0.3).to(torch.int32).to(self.dev)

    def _plain_routing(self, m, seed, E, topk):
        g = torch.Generator(device="cpu").manual_seed(seed)
        ids = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(m)])
        ids = ids.to(torch.int32).to(self.dev).contiguous()
        return ids, torch.rand(m, topk, generator=g).to(self.dev)

    def _sort_masked(self, fn, ids, w, bs, E, mask):
        return fn(
            ids,
            w,
            E,
            self.EP_DIM,
            torch.bfloat16,
            bs,
            expert_mask=mask,
            num_local_tokens=None,
            dispatch_policy=0,
            use_opus=True,
        )

    def _masks(self, E):
        return [(f"rank{r}", self._ep_mask(E, r, self.EP_SIZE)) for r in (0, 3, 7)] + [
            ("random", self._random_mask(E, E))
        ]

    def test_masked_sort_matches_aiter(self):
        E, topk = self.EP_E, self.EP_TOPK
        for bs in (16, 32, 64):
            for m in (1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 33, 64, 96, 128):
                ids, w = self._plain_routing(m, 100 + m, E, topk)
                for name, mask in self._masks(E):
                    self.assertTrue(
                        self.S._small_sort_supported(ids, bs, mask, None, E),
                        f"m={m} bs={bs} {name} not taken",
                    )
                    self._assert_same_sort(
                        self._sort_masked(self.orig_sort, ids, w, bs, E, mask),
                        self._sort_masked(
                            self.fm._moe_sorting_impl, ids, w, bs, E, mask
                        ),
                        bs,
                        f"E={E} m={m} bs={bs} {name}",
                    )

    def test_masked_sort_matches_aiter_with_shared_expert(self):
        # DeepSeek-V4.1 shape: 385 = 384 routed + fused shared, every rank owns the shared slot
        E, topk, bs = 385, 7, 32
        for m in (2, 4, 5, 10, 36, 37, 146):
            ids, w = self._routing(m, m, E, topk)
            for r in (0, 5):
                mask = self._ep_mask(E - 1, r, self.EP_SIZE)
                mask = torch.cat(
                    [mask, torch.ones(1, dtype=torch.int32, device=self.dev)]
                )
                self._assert_same_sort(
                    self._sort_masked(self.orig_sort, ids, w, bs, E, mask),
                    self._sort_masked(self.fm._moe_sorting_impl, ids, w, bs, E, mask),
                    bs,
                    f"E={E} m={m} rank{r}",
                )

    def test_masked_fused_mxfp8_quant_matches_aiter(self):
        E, topk, dim, bs = self.EP_E, self.EP_TOPK, self.EP_DIM, 32
        S = self.S
        for m in (1, 2, 4, 8, 16, 32, 33, 64, 128):
            ids, w = self._plain_routing(m, 200 + m, E, topk)
            x = torch.randn(m, dim, dtype=torch.bfloat16, device=self.dev)
            for name, mask in self._masks(E):
                sid, sw, _, nv, _ = self._sort_masked(
                    self.orig_sort, ids, w, bs, E, mask
                )
                ref_q, ref_s = self.orig_quant(
                    x,
                    sorted_ids=sid,
                    num_valid_ids=nv,
                    token_num=m,
                    topk=topk,
                    block_size=bs,
                    sorted_weights=sw,
                    num_experts_upper_bound=E,
                )
                tok = S._pending_quant_input.set(x)
                etok = S._emitted_quant.set(None)
                try:
                    psid = self._sort_masked(
                        self.fm._moe_sorting_impl, ids, w, bs, E, mask
                    )[0]
                    emitted = S._emitted_quant.get()
                finally:
                    S._emitted_quant.reset(etok)
                    S._pending_quant_input.reset(tok)
                msg = f"E={E} m={m} {name}"
                self.assertIsNotNone(emitted, msg)
                real = sid[: int(nv[0])]
                real = real[real != ((topk << 24) | m)]
                if real.numel() == 0:
                    continue
                # fp8 rows of the tokens this rank computes
                toks = torch.unique(real & 0xFFFFFF).long()
                self.assertTrue(
                    torch.equal(
                        ref_q.view(torch.uint8)[toks],
                        emitted[0].view(torch.uint8)[toks],
                    ),
                    msg,
                )

                # e8m0 bytes at each real sorted row: rows are matched by (slot, token)
                def rows_of(s):
                    s = s[: int(nv[0])]
                    rows = torch.nonzero(s != ((topk << 24) | m)).flatten()
                    order = torch.argsort(s[rows].long())
                    return rows[order]

                pad = (dim // 32 + 7) // 8 * 8
                g = torch.arange(dim // 32, device=self.dev)

                def scale_bytes(scales, rows):
                    addr = (rows // 32) * pad * 32 + rows % 16 * 4 + rows % 32 // 16
                    addr = addr[:, None] + (g // 8 * 256 + g % 4 * 64 + g % 8 // 4 * 2)
                    return scales.view(torch.uint8).flatten()[addr.flatten()]

                self.assertTrue(
                    torch.equal(
                        scale_bytes(ref_s, rows_of(sid)),
                        scale_bytes(emitted[1], rows_of(psid)),
                    ),
                    f"{msg} scales",
                )

    def test_masked_gate(self):
        E = self.EP_E
        mask = self._ep_mask(E, 0, self.EP_SIZE)
        for m in (1, 4, 8, 16, 32):
            ids, _ = self._plain_routing(m, m, E, self.EP_TOPK)
            self.assertTrue(self.S._small_sort_supported(ids, 32, mask, None, E), m)
        ids, _ = self._plain_routing(128, 128, E, self.EP_TOPK)
        self.assertTrue(self.S._small_sort_supported(ids, 32, mask, None, E))
        ids, _ = self._plain_routing(129, 129, E, self.EP_TOPK)
        self.assertFalse(self.S._small_sort_supported(ids, 32, mask, None, E))
        # a mask that does not cover every expert id goes back to aiter
        ids, _ = self._plain_routing(4, 4, E, self.EP_TOPK)
        self.assertFalse(self.S._small_sort_supported(ids, 32, mask[:-1], None, E))
        # 30 pairs at block 8 fit only the distributed kernel, which takes no mask
        ids, _ = self._plain_routing(6, 6, 129, 5)
        self.assertTrue(self.S._small_sort_supported(ids, 8, None, None, 129))
        mask = self._ep_mask(129, 0, self.EP_SIZE)
        self.assertFalse(self.S._small_sort_supported(ids, 8, mask, None, 129))


@unittest.skipUnless(_gfx950(), "gfx950 (MI35x) only")
class TestMoeSortingSmallPast256Pairs(CustomTestCase):
    """Every row count between 257 and 1024 pairs: the patched sort, its fused quant and
    stage 1 against aiter's, and the same patched outputs on every call."""

    # name, experts, topk, fused shared expert in the last slot, hidden
    SHAPES = (
        ("MiMo-V2.6-Pro", 384, 8, False, 6144),
        ("DeepSeek-V4.1", 385, 7, True, 5120),
    )
    BLOCK_SIZES = (16, 32, 64, 128)
    EP_SIZE = 8

    setUpClass = TestMoeSortingSmall.__dict__["setUpClass"]
    _ep_mask = TestMoeSortingSmallExpertMask._ep_mask
    _random_mask = TestMoeSortingSmallExpertMask._random_mask
    _mxfp4_experts = TestMoeSortingSmall._mxfp4_experts
    _stage1_out = TestMoeSortingSmall._stage1_out

    @staticmethod
    def _rows(topk):
        """row counts with more than 256 and at most 1024 pairs"""
        return range(256 // topk + 1, 1024 // topk + 1)

    def _masks(self, E, shared):
        """no mask, each EP8 rank's (every rank owns the shared expert), and a random one"""
        masks = [("none", None)]
        for r in range(self.EP_SIZE):
            if shared:
                mask = self._ep_mask(E - 1, r, self.EP_SIZE)
                mask = torch.cat([mask, mask.new_ones(1)])
            else:
                mask = self._ep_mask(E, r, self.EP_SIZE)
            masks.append((f"rank{r}", mask))
        return masks + [("random", self._random_mask(E, E))]

    def _ids(self, m, seed, E, topk, shared, pattern, mask=None):
        """uniform: any experts; hot: topk + 4 experts take every pair; same: every
        token on the same experts; offrank: only experts the mask leaves off this rank"""
        g = torch.Generator(device="cpu").manual_seed(seed)
        k, n = (topk - 1, E - 1) if shared else (topk, E)
        if pattern == "uniform":
            pool = torch.arange(n)
        elif pattern == "hot":
            pool = torch.randperm(n, generator=g)[: k + 4]
        elif pattern == "same":
            pool = torch.randperm(n, generator=g)[:k]
        else:
            pool = torch.nonzero(mask[:n].cpu() == 0).flatten()
        ids = pool[torch.rand(m, len(pool), generator=g).argsort(1)[:, :k]]
        if shared:
            ids = torch.cat([ids, torch.full((m, 1), E - 1)], 1)
        ids = ids.to(torch.int32).to(self.dev).contiguous()
        return ids, torch.rand(m, topk, generator=g).to(self.dev)

    def _sort(self, fn, ids, w, bs, E, mask, dim):
        return fn(ids, w, E, dim, torch.bfloat16, bs, mask, None, 0, True)

    def _emit(self, ids, w, x, bs, E, mask):
        """the patched sort with the quant input pending, and the quant it emitted"""
        S = self.S
        tok = S._pending_quant_input.set(x)
        etok = S._emitted_quant.set(None)
        try:
            out = self._sort(self.fm._moe_sorting_impl, ids, w, bs, E, mask, x.shape[1])
            return out, S._emitted_quant.get()
        finally:
            S._emitted_quant.reset(etok)
            S._pending_quant_input.reset(tok)

    def _assert_same_sort(self, a, b, bs, msg):
        self.assertTrue(torch.equal(a[3], b[3]), f"{msg} num_valid_ids")
        n = int(a[3][0])
        self.assertTrue(torch.equal(a[2][: n // bs], b[2][: n // bs]), f"{msg} experts")

        # the order of pairs inside one expert block is not part of the contract
        def blocks(s):
            ids = s[0][:n].to(torch.int64) << 32
            w = s[1][:n].view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            return (ids | w).view(-1, bs).sort(dim=1).values

        self.assertTrue(torch.equal(blocks(a), blocks(b)), f"{msg} pairs")
        self.assertTrue(bool((b[4] == 0).all()), f"{msg} moe_buf")

    @staticmethod
    def _real_rows(sid, nv, m, topk):
        """sorted rows that hold a pair, ordered by the packed (slot, token) id"""
        s = sid[: int(nv[0])]
        rows = torch.nonzero(s != ((topk << 24) | m)).flatten()
        return rows[torch.argsort(s[rows].long())]

    @staticmethod
    def _scale_bytes(scales, rows, dim):
        """e8m0 byte of every (row, 32-column group) at aiter's swizzled address"""
        pad, g = (dim // 32 + 7) // 8 * 8, torch.arange(dim // 32, device=rows.device)
        addr = (rows // 32) * pad * 32 + rows % 16 * 4 + rows % 32 // 16
        addr = addr[:, None] + (g // 8 * 256 + g % 4 * 64 + g % 8 // 4 * 2)
        return scales.view(torch.uint8).flatten()[addr.flatten()]

    def test_sort_matches_aiter_at_every_row_count(self):
        cases = 0
        for name, E, topk, shared, _ in self.SHAPES:
            rows = self._rows(topk)
            for bs in self.BLOCK_SIZES:
                for m in rows:
                    # uniform routing at every row count, the skewed patterns at every 8th
                    sparse = m in (rows[0], rows[-1]) or m % 8 == 0
                    for mname, mask in self._masks(E, shared):
                        patterns = ["uniform"]
                        if sparse:
                            patterns += ["hot", "same"]
                            patterns += [] if mask is None else ["offrank"]
                        for pattern in patterns:
                            ids, w = self._ids(m, m, E, topk, shared, pattern, mask)
                            msg = f"{name} m={m} bs={bs} {mname} {pattern}"
                            self.assertTrue(
                                self.S._small_sort_supported(ids, bs, mask, None, E),
                                f"{msg}: not taken",
                            )
                            self._assert_same_sort(
                                self._sort(self.orig_sort, ids, w, bs, E, mask, 1024),
                                self._sort(
                                    self.fm._moe_sorting_impl, ids, w, bs, E, mask, 1024
                                ),
                                bs,
                                msg,
                            )
                            cases += 1
        print(f"sort: {cases} cases match aiter")

    def test_fused_mxfp8_quant_matches_aiter(self):
        cases = 0
        for name, E, topk, shared, dim in self.SHAPES:
            rows = self._rows(topk)
            for m in sorted({rows[0], rows[-1], *rows[::6]}):
                x = torch.randn(m, dim, dtype=torch.bfloat16, device=self.dev)
                for bs in self.BLOCK_SIZES:
                    for mname, mask in self._masks(E, shared):
                        for pattern in ("uniform", "hot"):
                            ids, w = self._ids(m, m, E, topk, shared, pattern, mask)
                            msg = f"{name} m={m} bs={bs} {mname} {pattern}"
                            sid, sw, _, nv, _ = self._sort(
                                self.orig_sort, ids, w, bs, E, mask, dim
                            )
                            ref_q, ref_s = self.orig_quant(
                                x,
                                sorted_ids=sid,
                                num_valid_ids=nv,
                                token_num=m,
                                topk=topk,
                                block_size=bs,
                                sorted_weights=sw,
                                num_experts_upper_bound=E,
                            )
                            out, emitted = self._emit(ids, w, x, bs, E, mask)
                            self.assertIsNotNone(emitted, f"{msg}: quant not emitted")
                            ref_rows = self._real_rows(sid, nv, m, topk)
                            new_rows = self._real_rows(out[0], out[3], m, topk)
                            self.assertTrue(
                                torch.equal(sid[ref_rows], out[0][new_rows]), msg
                            )
                            # fp8 rows of the tokens this rank computes
                            toks = torch.unique(sid[ref_rows] & 0xFFFFFF).long()
                            self.assertTrue(
                                torch.equal(
                                    ref_q.view(torch.uint8)[toks],
                                    emitted[0].view(torch.uint8)[toks],
                                ),
                                f"{msg} fp8",
                            )
                            self.assertTrue(
                                torch.equal(
                                    self._scale_bytes(ref_s, ref_rows, dim),
                                    self._scale_bytes(emitted[1], new_rows, dim),
                                ),
                                f"{msg} scales",
                            )
                            cases += 1
        print(f"fused quant: {cases} cases match aiter")

    def test_same_output_on_every_call(self):
        cases = 0
        for name, E, topk, shared, dim in self.SHAPES:
            rows = self._rows(topk)
            masks = self._masks(E, shared)
            for m in (rows[0], rows[len(rows) // 2], rows[-1]):
                x = torch.randn(m, dim, dtype=torch.bfloat16, device=self.dev)
                for bs in self.BLOCK_SIZES:
                    for mname, mask in (masks[0], masks[1], masks[-2], masks[-1]):
                        for pattern in ("uniform", "hot"):
                            ids, w = self._ids(m, m, E, topk, shared, pattern, mask)
                            msg = f"{name} m={m} bs={bs} {mname} {pattern}"
                            runs = [
                                self._emit(ids, w, x, bs, E, mask) for _ in range(3)
                            ]
                            (sid, sw, eid, nv, _), (q, s) = runs[0]
                            n = int(nv[0])
                            real = self._real_rows(sid, nv, m, topk)
                            toks = torch.unique(sid[real] & 0xFFFFFF).long()
                            for (sid2, sw2, eid2, nv2, _), (q2, s2) in runs[1:]:
                                # positions too, not only the contract
                                self.assertTrue(torch.equal(nv, nv2), msg)
                                self.assertTrue(torch.equal(sid[:n], sid2[:n]), msg)
                                self.assertTrue(
                                    torch.equal(
                                        sw[:n].view(torch.int32),
                                        sw2[:n].view(torch.int32),
                                    ),
                                    msg,
                                )
                                self.assertTrue(
                                    torch.equal(eid[: n // bs], eid2[: n // bs]), msg
                                )
                                self.assertTrue(
                                    torch.equal(
                                        q.view(torch.uint8)[toks],
                                        q2.view(torch.uint8)[toks],
                                    ),
                                    f"{msg} fp8",
                                )
                                self.assertTrue(
                                    torch.equal(
                                        self._scale_bytes(s, real, dim),
                                        self._scale_bytes(s2, real, dim),
                                    ),
                                    f"{msg} scales",
                                )
                            cases += 1
        print(f"determinism: {cases} cases, 3 calls each, bitwise equal")

    def test_fused_moe_a8w4_stage1_matches_aiter(self):
        # MiMo-V2.6-Pro EP8 (48 local experts, inter 2048) and DeepSeek-V4.1 TP4 and EP8.
        # The last row count of each is one row past the limit: there the fused_moe
        # wrapper must not offer the quant and aiter sorts.
        cases = (
            (
                "MiMo-V2.6-Pro",
                384,
                8,
                False,
                6144,
                2048,
                0,
                None,
                (0, 5),
                (33, 48, 64, 96, 128, 129),
            ),
            (
                "DeepSeek-V4.1",
                385,
                7,
                True,
                5120,
                640,
                64,
                10.0,
                (None, 2),
                (37, 74, 110, 146, 147),
            ),
        )
        for name, E, topk, shared, dim, inter, pad, limit, ranks, rows in cases:
            masks = dict(self._masks(E, shared))
            for rank in ranks:
                mask = None if rank is None else masks[f"rank{rank}"]
                local = E if mask is None else int(mask.sum())
                experts = self._mxfp4_experts(local, hidden=dim, inter=inter)
                for m in rows:
                    for pattern in (
                        ("uniform", "hot") if m == rows[-2] else ("uniform",)
                    ):
                        ids, w = self._ids(m, m, E, topk, shared, pattern, mask)
                        x = torch.randn(m, dim, dtype=torch.bfloat16, device=self.dev)
                        args = (x, ids, w, experts)
                        kw = dict(expert_mask=mask, pad=pad, swiglu_limit=limit)
                        ref, _ = self._stage1_out(*args, stock=True, **kw)
                        out, fused = self._stage1_out(*args, stock=False, **kw)
                        msg = f"{name} rank={rank} m={m} {pattern}"
                        self.assertEqual(fused, m * topk <= 1024, f"{msg} fused")
                        # rows of (token, slot) pairs another rank computes stay unwritten
                        keep = torch.ones_like(ids, dtype=torch.bool)
                        if mask is not None:
                            keep = mask[ids.long()].bool()
                        keep = keep.flatten()
                        self.assertEqual(out.shape[0], m * topk, msg)
                        self.assertTrue(torch.equal(out[keep], ref[keep]), msg)
            del experts
            torch.cuda.empty_cache()

    def test_gate_at_the_pair_limit(self):
        self.assertEqual(self.S._MAX_PAIRS, 1024)
        for E, topk, shared in (
            (384, 8, False),
            (385, 7, True),
            (129, 8, False),
            (257, 9, False),
        ):
            hi = 1024 // topk
            for mname, mask in self._masks(E, shared)[:2]:
                for bs in self.BLOCK_SIZES:
                    ids, _ = self._ids(hi, hi, E, topk, shared, "uniform")
                    self.assertTrue(
                        self.S._small_sort_supported(ids, bs, mask, None, E),
                        f"E={E} m={hi} bs={bs} {mname}",
                    )
                    ids, _ = self._ids(hi + 1, hi, E, topk, shared, "uniform")
                    self.assertFalse(
                        self.S._small_sort_supported(ids, bs, mask, None, E),
                        f"E={E} m={hi + 1} bs={bs} {mname}",
                    )
        # exactly 1024 and 1025 pairs
        for m, topk, ok in (
            (128, 8, True),
            (1024, 1, True),
            (205, 5, False),
            (1025, 1, False),
        ):
            ids, _ = self._ids(m, m, 129, topk, False, "uniform")
            self.assertEqual(
                self.S._small_sort_supported(ids, 32, None, None, 129), ok, (m, topk)
            )


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "ROCm only")
class TestMoeSortingSmallNoQuant(CustomTestCase):
    """No-quant path of _run_small_sort (incl. the histogram + bitonic kernel for
    32 < P <= 1024) vs a torch reference of the aiter moe_sorting layout."""

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
            # every row count up to 256 pairs, then every 7th up to 1024
            rows = [
                *range(1, 256 // topk + 1),
                *range(256 // topk + 1, 1024 // topk + 1, 7),
            ]
            for m in rows + [1024 // topk]:
                for hot in (num_experts, 12):  # uniform and skewed routing
                    with self.subTest(e=num_experts, topk=topk, m=m, hot=hot):
                        self._check(m, topk, num_experts, block_size, hot)


if __name__ == "__main__":
    unittest.main()
