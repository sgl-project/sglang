"""DeepSeek-V4.1 ratio-1/2 on the fp8 two-pool unified_kv: writer, pool, decode streams.

The two-pool c1/c2 store runs the V4 paged store's quantization unchanged and only
moves the bytes, so every written row must match the paged V4 row byte for byte:
448 B fp8 nope, the 7 UE8M0 tile scales (each stored twice behind the payload) and
the 64 bf16 rope values in the second pool. Rows nobody wrote must stay zero.

The TP2 cases run the fused query/K store and the decode split merge at 32 local
heads (TP4 runs 16). The DSpark cases cover target verify and the draft block window.
"""

import contextlib
import unittest

import torch

from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
    c1_decode_norm_rope_store,
    c2_decode_norm_rope_store,
    c2_prefill_norm_rope_store,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime
from sglang.kernels.ops.attention.fused_qk_norm_rope_store import pack_q_fp8_2buff
from sglang.kernels.ops.speculative.dspark.dspark_attn_metadata import (
    BuildDsparkSwaRingIndices,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4UnifiedKVPool,
    get_compress_state_ring_size,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# the two-pool fp8 layout is only allocated on gfx95
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

_needs_gfx950 = unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "two-pool fp8 unified KV is gfx95-only",
)

DEV = torch.device("cuda")
HEAD, ROPE, NOPE = 512, 64, 448
EPS = 1e-6
V4_DATA, V4_SCALE = 576, 8


def v4_page_bytes(page: int) -> int:
    return -(-page * (V4_DATA + V4_SCALE) // 576) * 576


def paged_cache(page: int, max_slot: int) -> torch.Tensor:
    npages = max_slot // page + 2
    return torch.zeros(npages, v4_page_bytes(page), dtype=torch.uint8, device=DEV)


def paged_row(cache: torch.Tensor, page: int, slot: int):
    p, o = divmod(slot, page)
    data = cache[p, o * V4_DATA : (o + 1) * V4_DATA]
    scale = cache[p, page * V4_DATA + o * V4_SCALE : page * V4_DATA + o * V4_SCALE + 7]
    return data[:NOPE], data[NOPE:], scale


def two_pool(rows: int):
    nope = torch.zeros(rows, 512, dtype=torch.float8_e4m3fn, device=DEV)
    rope = torch.zeros(rows, ROPE, dtype=torch.bfloat16, device=DEV)
    return nope, rope


def freqs(max_pos: int) -> torch.Tensor:
    return torch.randn(max_pos, ROPE, dtype=torch.float32, device=DEV)


class _Base(CustomTestCase):
    def assert_rows_match(self, paged, page, nope_pool, rope_pool, row_base, slots):
        nb = nope_pool.view(torch.uint8)
        rb = rope_pool.view(torch.uint8)
        written = torch.zeros(nb.shape[0], dtype=torch.bool, device=DEV)
        for s in slots:
            p_nope, p_rope, p_scale = paged_row(paged, page, s)
            r = row_base + s
            written[r] = True
            self.assertTrue(torch.equal(nb[r, :NOPE], p_nope), f"nope slot {s}")
            self.assertTrue(
                torch.equal(nb[r, NOPE : NOPE + 14], p_scale.repeat_interleave(2)),
                f"scales slot {s}",
            )
            self.assertTrue(torch.equal(rb[r], p_rope), f"rope slot {s}")
            self.assertTrue(bool(p_scale.ne(0).all()), f"paged slot {s} never written")
        # nothing outside the written rows (pad bytes of written rows included)
        self.assertEqual(int(nb[~written].count_nonzero()), 0)
        self.assertEqual(int(rb[~written].count_nonzero()), 0)
        self.assertEqual(int(nb[written, NOPE + 14 :].count_nonzero()), 0)


@_needs_gfx950
class TestC1(_Base):
    def test_matches_paged(self):
        torch.manual_seed(1)
        page, n, row_base = 256, 9, 777
        kv = torch.randn(n, HEAD, dtype=torch.bfloat16, device=DEV) * 3
        w = torch.randn(HEAD, dtype=torch.bfloat16, device=DEV)
        pos = torch.randint(0, 4000, (n,), device=DEV)
        loc = torch.randperm(3000, device=DEV)[:n] + 1
        loc[4] = 0  # padded graph row: computes, stores nothing
        fc = freqs(4096)
        paged = paged_cache(page, 3001)
        out_p = c1_decode_norm_rope_store(
            kv, w, pos, loc, EPS, fc, paged, page_size=page
        )
        nope, rope = two_pool(row_base + 3001 + page)
        out_u = c1_decode_norm_rope_store(
            kv,
            w,
            pos,
            loc,
            EPS,
            fc,
            nope,
            page_size=page,
            fp8_2buff=True,
            k_cache_rope=rope,
            row_base=row_base,
        )
        self.assertTrue(torch.equal(out_p, out_u))
        slots = [int(x) for x in loc.tolist() if x > 0]
        self.assert_rows_match(paged, page, nope, rope, row_base, slots)


@_needs_gfx950
class TestC2(_Base):
    page = 128
    ring = 2

    def _common(self, n_req):
        torch.manual_seed(2)
        w = torch.randn(HEAD, dtype=torch.bfloat16, device=DEV)
        state = torch.randn(
            n_req * self.ring, 2 * HEAD, dtype=torch.float32, device=DEV
        )
        return w, state, freqs(4096)

    def test_decode_matches_paged(self):
        n_req, row_base = 5, 1234
        w, state0, fc = self._common(n_req)
        req = torch.arange(n_req, device=DEV, dtype=torch.int64)
        base_pos = torch.tensor([0, 10, 100, 2000, 64], device=DEV)
        full = torch.randperm(1000, device=DEV)[:n_req] * 2 + 2  # even FULL slots
        steps = []
        for k in range(2):  # even position parks, odd one completes the pair
            kv = torch.randn(n_req, 2 * HEAD, dtype=torch.float32, device=DEV)
            pos = base_pos + k
            loc = full + k
            steps.append((kv, pos, loc))
        steps[1][2][3] = 0  # a padded row on the completing step

        def run(two):
            state = state0.clone()
            if two:
                nope, rope = two_pool(row_base + 1100 + self.page)
                cache, kw = (
                    nope,
                    dict(fp8_2buff=True, k_cache_rope=rope, row_base=row_base),
                )
            else:
                cache, kw = paged_cache(self.page, 1100), {}
            outs = []
            for kv, pos, loc in steps:
                # rows that park or pad leave kv_output alone
                out = torch.zeros(n_req, HEAD, dtype=torch.bfloat16, device=DEV)
                outs.append(
                    c2_decode_norm_rope_store(
                        kv,
                        state,
                        w,
                        pos,
                        req,
                        loc,
                        EPS,
                        fc,
                        cache,
                        page_size=self.page,
                        ring_size=self.ring,
                        out=out,
                        **kw,
                    )
                )
            return cache, kw, state, outs

        paged, _, st_p, out_p = run(False)
        nope, kw, st_u, out_u = run(True)
        self.assertTrue(torch.equal(st_p, st_u))
        self.assertTrue(torch.equal(out_p[1], out_u[1]))
        slots = [int(x) >> 1 for x in steps[1][2].tolist() if x > 0]
        self.assert_rows_match(
            paged, self.page, nope, kw["k_cache_rope"], row_base, slots
        )

    def test_prefill_matches_paged(self):
        row_base = 99
        w, state0, fc = self._common(3)
        # req 0: positions 0..10; req 1: 5..16 (starts odd -> partner from the ring);
        # req 2: 31..33
        spans = [(0, 0, 11), (1, 5, 12), (2, 31, 3)]
        req, pos, starts, lens = [], [], [], []
        for r, p0, ln in spans:
            starts.append(len(pos))
            lens.append(ln)
            req += [r] * ln
            pos += list(range(p0, p0 + ln))
        n = len(pos)
        req = torch.tensor(req, device=DEV, dtype=torch.int64)
        pos = torch.tensor(pos, device=DEV, dtype=torch.int64)
        # FULL slots: request-contiguous, aligned so slot = loc >> 1 pairs within a request
        loc = torch.cat(
            [torch.arange(p0, p0 + ln) + 400 * (r + 1) for r, p0, ln in spans]
        ).to(DEV)
        kv = torch.randn(n, 2 * HEAD, dtype=torch.float32, device=DEV)
        start = torch.tensor(starts, device=DEV, dtype=torch.int32)
        seqlen = torch.tensor(lens, device=DEV, dtype=torch.int32)

        def run(two):
            state = state0.clone()
            if two:
                nope, rope = two_pool(row_base + 1300 + self.page)
                cache, kw = (
                    nope,
                    dict(fp8_2buff=True, k_cache_rope=rope, row_base=row_base),
                )
            else:
                cache, kw = paged_cache(self.page, 1300), {}
            out = torch.zeros(n, HEAD, dtype=torch.bfloat16, device=DEV)
            c2_prefill_norm_rope_store(
                kv,
                state,
                w,
                pos,
                req,
                loc,
                EPS,
                fc,
                cache,
                start,
                seqlen,
                page_size=self.page,
                ring_size=self.ring,
                out=out,
                **kw,
            )
            return cache, kw, state, out

        paged, _, st_p, out_p = run(False)
        nope, kw, st_u, out_u = run(True)
        self.assertTrue(torch.equal(st_p, st_u))
        self.assertTrue(torch.equal(out_p, out_u))
        odd = pos % 2 == 1
        slots = sorted({int(x) >> 1 for x in loc[odd].tolist()})
        self.assert_rows_match(
            paged, self.page, nope, kw["k_cache_rope"], row_base, slots
        )


class _Adapter:
    def region(self, *_a, **_k):
        return contextlib.nullcontext()


@_needs_gfx950
class TestPool(CustomTestCase):
    def test_kv_source_groups(self):
        # V4.1-Flash: 0 0 | 2 x18 | 1 x20 | 0 0 0, sources 2 8 14 20
        ratios = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
        sources = [2, 8, 14, 20]
        group_of = []
        for i, r in enumerate(ratios):
            group_of.append(max(s for s in sources if s <= i) if r in (1, 2) else None)
        num_slots, ring, num_blocks, page = 3, 128, 6, 256
        pool = DeepSeekV4UnifiedKVPool(
            stage_ratios=ratios,
            num_slots=num_slots,
            num_blocks=num_blocks,
            page_size=page,
            qk_nope_head_dim=NOPE,
            qk_rope_head_dim=ROPE,
            device=DEV,
            memory_saver_adapter=_Adapter(),
            custom_mem_pool=None,
            swa_ring_size=ring,
            fp8=True,
            group_of=group_of,
        )
        swa = num_slots * ring
        for i, r in enumerate(ratios):
            buf, rope = pool.kv_buffer[i], pool.kv_buffer_rope[i]
            self.assertEqual(buf.shape[0], rope.shape[0])
            self.assertTrue(buf.is_contiguous() and rope.is_contiguous())
            base = pool.get_compress_row_base(i)
            if r == 0:
                self.assertEqual(base, swa)
                self.assertEqual(buf.shape[0], swa)
                continue
            src = group_of[i]
            members = [k for k, g in enumerate(group_of) if g == src]
            j, G = members.index(i), len(members)
            self.assertEqual(base, (G - j) * swa)
            compress = num_blocks * (128 // r) + page // r
            self.assertEqual(buf.shape[0], base + compress)
            # same compressed region as the source, distinct SWA ring
            src_buf = pool.kv_buffer[src]
            self.assertEqual(
                buf[base:].data_ptr(),
                src_buf[pool.get_compress_row_base(src) :].data_ptr(),
            )
            if i != src:
                self.assertNotEqual(buf.data_ptr(), src_buf.data_ptr())
                self.assertEqual(buf.data_ptr() - src_buf.data_ptr(), j * swa * 512)
        self.assertEqual(len(pool.groups), 4)


def _swa_prefix_stream(state_slot, positions, swa_len, compress_len, ring):
    """Per-layer decode stream with only its SWA-ring prefix filled; the compressed
    tail is left for fill_compress_tail."""
    lengths = (swa_len + compress_len).to(torch.int64)
    indptr = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=DEV)
    indptr[1:] = torch.cumsum(lengths, 0)
    indices = torch.full((int(indptr[-1]),), -1, dtype=torch.int32, device=DEV)
    for t in range(lengths.shape[0]):
        a, n, pos = int(indptr[t]), int(swa_len[t]), int(positions[t])
        rows = torch.arange(pos - n + 1, pos + 1, device=DEV) % ring
        indices[a : a + n] = (int(state_slot[t]) * ring + rows).int()
    return indices, indptr


@_needs_gfx950
class TestDecodeStream(CustomTestCase):
    def test_group_streams_match_per_layer(self):
        torch.manual_seed(7)
        win, ring, topk, swa_pages = 128, 128, 512, 4096
        for N, G in ((1, 4), (5, 6), (16, 4)):
            state_slot = torch.randperm(32, device=DEV)[:N].int()
            positions = torch.randint(0, 5000, (N,), device=DEV, dtype=torch.int64)
            positions[0] = 3  # a short request: SWA prefix shorter than the window
            swa_len = torch.clamp(positions + 1, max=win).int()
            for ratio in (1, 2):
                low_len = torch.clamp((positions + 1) // ratio, max=topk).int()
                page_idx = torch.full((N, topk), -1, dtype=torch.int32, device=DEV)
                for t in range(N):
                    k = int(low_len[t])
                    page_idx[t, :k] = torch.randperm(20000, device=DEV)[:k].int()
                    if k > 2:
                        page_idx[t, 1] = -1  # an empty top-k slot in the middle
                indptr = runtime.low_ratio_decode_indptr(swa_len, low_len)
                base0 = (G + 3) * swa_pages
                got = runtime.build_low_ratio_group_streams(
                    state_slot=state_slot,
                    positions=positions,
                    indptr=indptr,
                    prefix_len=swa_len,
                    page_indices=page_idx,
                    valid_len=low_len,
                    num_layers=G,
                    base0=base0,
                    base_step=-swa_pages,
                    win=win,
                    ring_stride=ring,
                )
                for g in range(G):
                    ref, ref_p = _swa_prefix_stream(
                        state_slot, positions, swa_len, low_len, ring
                    )
                    runtime.fill_compress_tail(
                        indices=ref,
                        indptr=ref_p,
                        prefix_len=swa_len,
                        page_indices=page_idx,
                        valid_len=low_len,
                        swa_pages=base0 - g * swa_pages,
                    )
                    self.assertTrue(torch.equal(indptr, ref_p))
                    n = int(ref_p[-1])
                    self.assertTrue(
                        torch.equal(got[g, :n], ref[:n]), f"{N=} {ratio=} {g=}"
                    )

    def test_v41_meta_matches_generic(self):
        torch.manual_seed(8)
        win, ring, topk, swa_pages = 128, 136, 512, 4096
        for N in (1, 3, 16, 37):
            state_slot = torch.randperm(64, device=DEV)[
                :N
            ]  # int64 like req_pool_indices
            positions = torch.randint(0, 3000, (N,), device=DEV, dtype=torch.int32)
            positions[0] = 0
            seq_lens = positions + 1
            swa_len = torch.clamp(positions + 1, max=win).int()
            got = runtime.build_v41_decode_meta(
                state_slot=state_slot,
                positions=positions,
                seq_lens=seq_lens,
                swa_len=swa_len,
                ratios=(1, 2),
                topk=topk,
                win=win,
                ring_stride=ring,
            )
            zeros = torch.zeros_like(swa_len)
            swa_i, swa_p, *_ = runtime.build_decode_streams(
                state_slot=state_slot,
                positions=positions,
                swa_len=swa_len,
                hca_len=zeros,
                csa_len=zeros,
                hca_page_indices=torch.empty((N, 0), dtype=torch.int32, device=DEV),
                csa_width=0,
                win=win,
                ring_stride=ring,
                swa_pages=swa_pages,
            )
            self.assertTrue(torch.equal(got["swa_indptr"], swa_p))
            n = int(swa_p[-1])
            self.assertTrue(torch.equal(got["swa_indices"][:n], swa_i[:n]))
            for r in (1, 2):
                low_len = torch.clamp(seq_lens.to(torch.int32) // r, max=topk)
                self.assertTrue(torch.equal(got[f"c{r}_len"], low_len))
                self.assertTrue(
                    torch.equal(
                        got[f"c{r}_indptr"],
                        runtime.low_ratio_decode_indptr(swa_len, low_len),
                    )
                )
            loc = state_slot.to(torch.int64) * ring + positions.to(torch.int64) % ring
            self.assertTrue(torch.equal(got["swa_loc_i64"], loc))
            self.assertTrue(torch.equal(got["swa_loc"], loc.to(torch.int32)))

    def test_prefill_swa_floor(self):
        """Decoder SWA bounded replay: build_prefill_indices with swa_floor drops the
        SWA rows below each token's floor (prefix and extend), keeps the compressed
        rows, and without a floor is unchanged."""
        torch.manual_seed(9)
        win, ring, topk, base = 128, 136, 64, 7000
        # two requests: a 300-token tail chunk at 5000 and a 40-token chunk at 90
        chunks = [(5000, 300, 3), (90, 40, 5)]
        pos, cst, cuq, slot = [], [], [], []
        off = 0
        for start, n, s_ in chunks:
            for i in range(n):
                pos.append(start + i)
                cst.append(start)
                cuq.append(off)
                slot.append(s_)
            off += n
        T = len(pos)
        mk = lambda v: torch.tensor(v, dtype=torch.int64, device=DEV)
        positions, chunk_start, cu_q, state_slot = mk(pos), mk(cst), mk(cuq), mk(slot)
        page_idx = torch.full((T, topk), -1, dtype=torch.int32, device=DEV)
        for t in range(T):
            k = int(torch.randint(0, topk + 1, (1,)))
            page_idx[t, :k] = torch.randperm(5000, device=DEV)[:k].int()
        kw = dict(
            compress_ratio=2,
            state_slot=state_slot,
            positions=positions,
            chunk_start=chunk_start,
            cu_q=cu_q,
            win=win,
            ring_stride=ring,
            swa_pages=base,
            c128_page_indices=None,
            c4_sparse_page_indices=None,
            low_ratio_page_indices=page_idx,
        )
        plain = runtime.build_prefill_indices(**kw)
        zero = runtime.build_prefill_indices(
            **kw, swa_floor=torch.zeros(T, dtype=torch.int32, device=DEV)
        )
        # the index buffers are torch.empty past their indptr ends
        for i, p_ in ((0, 1), (2, 3)):
            self.assertTrue(torch.equal(plain[p_], zero[p_]))
            n = int(plain[p_][-1])
            self.assertTrue(torch.equal(plain[i][:n], zero[i][:n]))
        # floors: the tail start (the replay case), one inside the prior window, one past cstart
        floors = []
        for t in range(T):
            floors.append([cst[t], cst[t] - 50, cst[t] + 10][t % 3])
        pre_i, pre_p, ext_i, ext_p = runtime.build_prefill_indices(
            **kw, swa_floor=torch.tensor(floors, dtype=torch.int32, device=DEV)
        )
        for t in range(T):
            p_, c_, f_ = pos[t], cst[t], floors[t]
            low = max(p_ - win + 1, 0, f_)
            want_pre = [slot[t] * ring + g % ring for g in range(low, c_)]
            want_pre += [int(x) + base for x in page_idx[t].tolist() if x >= 0]
            want_ext = [cuq[t] + (g - c_) for g in range(max(c_, low), p_ + 1)]
            got_pre = pre_i[int(pre_p[t]) : int(pre_p[t + 1])].tolist()
            got_ext = ext_i[int(ext_p[t]) : int(ext_p[t + 1])].tolist()
            self.assertEqual(got_pre, want_pre, t)
            self.assertEqual(got_ext, want_ext, t)


@_needs_gfx950
class TestTP2Heads(_Base):
    def test_fused_store_h32(self):
        from sglang.kernels.ops.attention.dsv4 import (
            fused_k_norm_rope_flashmla,
            fused_qk_norm_rope_fp8_2buff,
        )

        torch.manual_seed(11)
        for T in (1, 9, 64):
            H = 32
            q = torch.randn(T, H, HEAD, dtype=torch.bfloat16, device=DEV) * 2
            kv = torch.randn(T, 600, dtype=torch.bfloat16, device=DEV)[
                :, 40 : 40 + HEAD
            ]
            w = torch.randn(HEAD, dtype=torch.bfloat16, device=DEV)
            pos = torch.randint(0, 4096, (T,), device=DEV, dtype=torch.int64)
            ang = torch.rand(4096, ROPE // 2, device=DEV) * 6.28
            fc = torch.polar(torch.ones_like(ang), ang)
            rows, page = 200, 64
            loc = torch.randperm(rows, device=DEV)[:T].int()
            paged = paged_cache(page, rows)
            q_ref = q.clone()
            fused_k_norm_rope_flashmla(
                kv, w, EPS, fc, pos, loc, paged, page_size=page, q=q_ref
            )
            nope, rope = two_pool(rows)
            qo = torch.empty(T, H, 512, dtype=torch.float8_e4m3fn, device=DEV)
            qr = torch.empty(T, H, ROPE, dtype=torch.bfloat16, device=DEV)
            fused_qk_norm_rope_fp8_2buff(
                q, kv, w, EPS, fc, pos, loc, nope, rope, qo, qr
            )
            torch.cuda.synchronize()
            self.assert_rows_match(
                paged, page, nope, rope, 0, [int(x) for x in loc.tolist()]
            )
            self.assertTrue(torch.equal(qr, q_ref[..., NOPE:]))
            po, pr = torch.empty_like(qo), torch.empty_like(qr)
            pack_q_fp8_2buff(q, po, pr, fc, pos)
            self.assertTrue(
                torch.equal(
                    qo.view(torch.uint8)[..., :462], po.view(torch.uint8)[..., :462]
                )
            )

    def test_decode_merge_h32(self):
        self._merge(32)

    def test_decode_merge_h16_same_shapes(self):
        self._merge(16)

    def _merge(self, H):
        torch.manual_seed(12)
        rows = 50_000

        def packed(n, h):
            x = torch.randn(n, h, HEAD, dtype=torch.bfloat16, device=DEV)
            a = torch.empty(n, h, 512, dtype=torch.float8_e4m3fn, device=DEV)
            b = torch.empty(n, h, ROPE, dtype=torch.bfloat16, device=DEV)
            pack_q_fp8_2buff(x, a, b)
            return a, b

        kvn, kvr = packed(rows, 1)
        kvn, kvr = kvn.view(rows, 512), kvr.view(rows, ROPE)
        sink = torch.randn(H, dtype=torch.float32, device=DEV)
        ang = torch.rand(8192, ROPE // 2, device=DEV) * 6.28
        fr = torch.view_as_real(torch.polar(torch.ones_like(ang), ang)).flatten(-2)
        for T, L, splits in ((1, 640, 8), (8, 640, 8), (64, 640, None), (5, 128, 1)):
            q, qr = packed(T, H)
            lens = torch.randint(L // 2, L + 1, (T,), device=DEV)
            indptr = torch.zeros(T + 1, dtype=torch.int32, device=DEV)
            indptr[1:] = torch.cumsum(lens, 0)
            idx = torch.randint(
                0, rows, (int(indptr[-1]),), dtype=torch.int32, device=DEV
            )
            pos = torch.randint(0, 8192, (T,), dtype=torch.int64, device=DEV)
            kw = dict(
                q=q,
                q_rope=qr,
                unified_kv=kvn,
                unified_kv_rope=kvr,
                kv_indices=idx,
                kv_indptr=indptr,
                attn_sink=sink,
                v_head_dim=HEAD,
                num_kv_splits=splits,
                compress_ratio=1,
                low_ratio_model=True,
            )
            ref = runtime.decode_fp8_2buff(**kw).clone()
            runtime.apply_inverse_rope(ref, (fr, pos))
            got = runtime.decode_fp8_2buff(**kw, inv_rope=(fr, pos))
            self.assertFalse(bool(torch.isnan(got).any()), (T, L, splits))
            diff = (got.float() - ref.float()).abs()
            ulp = ref.float().abs().clamp_min(1e-30) * 2**-7
            r = diff / ulp
            # the reference rounds to bf16 before its RoPE pass, the merge ropes in fp32: the
            # rope tail may differ by a few ulps; the nope part must stay within one
            print(
                f"   H={H} T={T}: nope max {float(r[..., :NOPE].max()):.2f} ulp, rope max {float(r[..., NOPE:].max()):.2f} ulp, "
                f"rope exact {float((diff[..., NOPE:] == 0).float().mean()):.4f}, abs max {float(diff.max()):.2e}"
            )
            # accumulation-order noise: within one bf16 ulp of the reference or 2e-4 absolute
            # (near-zero outputs, where a relative ulp is meaningless); same at H=16 and H=32
            self.assertTrue(
                bool((diff <= torch.maximum(ulp, torch.full_like(ulp, 2e-4))).all()),
                (H, T, L, splits),
            )
            print(f"H={H} T={T} L={L} splits={splits}: merge matches stage2 + rope")


@_needs_gfx950
class TestNoQNormStore(CustomTestCase):
    """DeepSeek-V4.1's query has no head norm: K through aiter's K-only op must equal
    the K side of the fused Q/K op V4 uses; Q is only packed."""

    def _inputs(self, T=7, H=16):
        torch.manual_seed(4)
        q = torch.randn(T, H, HEAD, dtype=torch.bfloat16, device=DEV) * 2
        base = torch.randn(T, 600, dtype=torch.bfloat16, device=DEV)
        kv = base[:, 40 : 40 + HEAD]  # strided like the qkv_a slice
        w = torch.randn(HEAD, dtype=torch.bfloat16, device=DEV)
        pos = torch.randint(0, 3000, (T,), device=DEV, dtype=torch.int64)
        ang = torch.randn(4096, ROPE // 2, device=DEV)
        return q, kv, w, pos, ang.cos().bfloat16(), ang.sin().bfloat16()

    def test_pack_q(self):
        from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
            pack_q_fp8_2buff,
        )

        q, *_ = self._inputs()
        T, H, _ = q.shape
        q_out = torch.empty(T, H, 512, dtype=torch.float8_e4m3fn, device=DEV)
        q_rope = torch.empty(T, H, ROPE, dtype=torch.bfloat16, device=DEV)
        pack_q_fp8_2buff(q, q_out, q_rope)
        self.assertTrue(torch.equal(q_rope, q[..., NOPE:]))
        b = q_out.view(torch.uint8)
        sc = b[..., NOPE : NOPE + 14].view(T, H, 7, 2)
        self.assertTrue(torch.equal(sc[..., 0], sc[..., 1]))
        scale = torch.exp2(sc[..., 0].float() - 127)  # [T, H, 7]
        deq = q_out[..., :NOPE].float().view(T, H, 7, 64) * scale[..., None]
        ref = q[..., :NOPE].float().view(T, H, 7, 64)
        amax = ref.abs().amax(-1, keepdim=True)
        # e4m3: 3 mantissa bits -> <= 2^-4 relative to the tile's power-of-two scale
        self.assertLessEqual(float(((deq - ref).abs() / amax).max()), 0.0626)
        # ceil to a power of two never clips: every scaled value fits in 448
        self.assertTrue(bool((amax.squeeze(-1) / scale <= 448).all()))

    def test_k_matches_fused_qk(self):
        from aiter.ops.fused_qk_norm_rope_cache_quant import (
            fused_qk_norm_rope_group_quant,
        )

        from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
            fused_qk_norm_rope_swa_store,
        )

        q, kv, w, pos, cos, sin = self._inputs()
        T, H, _ = q.shape
        rows = 64
        swa_loc = torch.randperm(rows, device=DEV)[:T].int()

        # reference: the fused Q/K op as _fp8_2buff_store calls it (V4)
        ref_nope, ref_rope = two_pool(rows)
        ref_kn = torch.zeros(T, 512, dtype=torch.float8_e4m3fn, device=DEV)
        ref_kr = torch.zeros(T, ROPE, dtype=torch.bfloat16, device=DEV)
        fused_qk_norm_rope_group_quant(
            q.clone(),
            kv,
            w,
            pos,
            cos,
            sin,
            EPS,
            is_neox=False,
            q_nope_scale_buff=torch.empty(
                T, H, 512, dtype=torch.float8_e4m3fn, device=DEV
            ),
            q_rope_buff=torch.empty(T, H, ROPE, dtype=torch.bfloat16, device=DEV),
            k_nope_scale_buff=ref_kn,
            k_rope_buff=ref_kr,
            quant_group_size=64,
            scale_dtype="e8m0",
            swa_nope_scale_buff=ref_nope,
            swa_rope_buff=ref_rope,
            swa_dest_row=swa_loc,
            batch_id_per_token=torch.arange(T, dtype=torch.int32, device=DEV),
        )

        nope, rope = two_pool(rows)
        kn = torch.zeros(T, 512, dtype=torch.float8_e4m3fn, device=DEV)
        kr = torch.zeros(T, ROPE, dtype=torch.bfloat16, device=DEV)
        q_out = torch.empty(T, H, 512, dtype=torch.float8_e4m3fn, device=DEV)
        q_rope = torch.empty(T, H, ROPE, dtype=torch.bfloat16, device=DEV)
        for kwargs in (
            dict(swa_cache=nope, swa_rope_cache=rope, swa_loc=swa_loc),
            dict(k_nope_out=kn, k_rope_out=kr),
        ):
            fused_qk_norm_rope_swa_store(
                q=q,
                kv=kv,
                q_norm_weight=None,
                kv_norm_weight=w,
                q_rms_eps=EPS,
                kv_rms_eps=EPS,
                rope_head_dim=ROPE,
                cos_cache=cos,
                sin_cache=sin,
                positions=pos,
                swa_page_size=1,
                q_out=q_out,
                fp8_2buff=True,
                q_rope_out=q_rope,
                q_head_norm=False,
                **kwargs,
            )
        u8 = lambda t: t.view(torch.uint8)
        # payload + scales; the 50 B pad is never read and not part of the contract
        self.assertTrue(torch.equal(u8(nope)[:, :462], u8(ref_nope)[:, :462]))
        self.assertTrue(torch.equal(rope, ref_rope))
        self.assertTrue(torch.equal(u8(kn)[:, :462], u8(ref_kn)[:, :462]))
        self.assertTrue(torch.equal(kr, ref_kr))
        self.assertTrue(torch.equal(q_rope, q[..., NOPE:]))
        self.assertGreater(int(u8(nope)[swa_loc.long()].count_nonzero()), 0)


@_needs_gfx950
class TestFusedQKTwoPool(_Base):
    """The one-launch decode store: its K rows must be the paged V4 store's rows byte for
    byte (moved), its packed query the Triton packer's nope bytes, and its query rope
    bitwise the rope the paged path's K launch applies to the query."""

    def test_matches_paged_k_and_packer_q(self):
        from sglang.kernels.ops.attention.dsv4 import (
            fused_k_norm_rope_flashmla,
            fused_qk_norm_rope_fp8_2buff,
        )
        from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
            pack_q_fp8_2buff,
        )

        torch.manual_seed(6)
        for T, H in ((1, 16), (7, 16), (16, 16), (5, 8)):
            q = torch.randn(T, H, HEAD, dtype=torch.bfloat16, device=DEV) * 2
            base = torch.randn(T, 600, dtype=torch.bfloat16, device=DEV)
            kv = base[:, 40 : 40 + HEAD]  # strided like the qkv_a slice
            w = torch.randn(HEAD, dtype=torch.bfloat16, device=DEV)
            pos = torch.randint(0, 4096, (T,), device=DEV, dtype=torch.int64)
            ang = torch.rand(4096, ROPE // 2, device=DEV) * 6.28
            fc = torch.polar(torch.ones_like(ang), ang)
            rows, page = 96, 64
            loc = torch.randperm(rows, device=DEV)[:T].int()
            if T > 1:
                loc[1] = -1  # skipped: no ring row
            slots = [int(x) for x in loc.tolist() if x >= 0]

            paged = paged_cache(page, rows)
            q_ref = q.clone()
            fused_k_norm_rope_flashmla(
                kv, w, EPS, fc, pos, loc, paged, page_size=page, q=q_ref
            )
            nope, rope = two_pool(rows)
            q_out = torch.empty(T, H, 512, dtype=torch.float8_e4m3fn, device=DEV)
            q_rope = torch.empty(T, H, ROPE, dtype=torch.bfloat16, device=DEV)
            fused_qk_norm_rope_fp8_2buff(
                q, kv, w, EPS, fc, pos, loc, nope, rope, q_out, q_rope
            )
            torch.cuda.synchronize()
            self.assert_rows_match(paged, page, nope, rope, 0, slots)
            # query: the paged launch's rope bitwise, the packer's quant bytewise
            self.assertTrue(torch.equal(q_rope, q_ref[..., NOPE:]), f"q rope {T=}")
            po = torch.empty_like(q_out)
            pr = torch.empty_like(q_rope)
            pack_q_fp8_2buff(q, po, pr, fc, pos)
            u8 = lambda t: t.view(torch.uint8)
            self.assertTrue(
                torch.equal(u8(q_out)[..., :462], u8(po)[..., :462]), f"q nope {T=}"
            )
            print(f"T={T} H={H}: K rows match paged V4, Q nope matches packer")


@_needs_gfx950
class TestPackQRope(CustomTestCase):
    def test_rope_in_packer_matches_rope_then_pack(self):
        from sglang.kernels.ops.attention.dsv4.rope import (
            apply_rotary_emb_triton,
            set_batched_rope,
        )
        from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
            pack_q_fp8_2buff,
        )

        set_batched_rope(True)  # what DeepseekV4 runs on HIP
        torch.manual_seed(5)
        T, H = 9, 16
        q = torch.randn(T, H, HEAD, dtype=torch.bfloat16, device=DEV) * 2
        ang = torch.rand(4096, ROPE // 2, device=DEV) * 6.28
        freqs = torch.polar(torch.ones_like(ang), ang)  # complex64
        pos = torch.randint(0, 4096, (T,), device=DEV, dtype=torch.int64)

        ref_q = q.clone()
        apply_rotary_emb_triton(ref_q[..., NOPE:], freqs, positions=pos)
        mk = lambda: (
            torch.empty(T, H, 512, dtype=torch.float8_e4m3fn, device=DEV),
            torch.empty(T, H, ROPE, dtype=torch.bfloat16, device=DEV),
        )
        ro, rr = mk()
        pack_q_fp8_2buff(ref_q, ro, rr)
        uo, ur = mk()
        pack_q_fp8_2buff(q, uo, ur, freqs, pos)
        u8 = lambda t: t.view(torch.uint8)
        self.assertTrue(torch.equal(u8(uo)[..., :462], u8(ro)[..., :462]))
        diff = (ur.float() - rr.float()).abs()
        ulp = rr.float().abs().clamp_min(1e-30) * 2**-7
        self.assertTrue(bool((diff <= ulp).all()), float((diff / ulp).max()))
        print("rope bit-exact fraction", float((diff == 0).float().mean()))


@_needs_gfx950
class TestDecodeInvRope(CustomTestCase):
    """Inverse RoPE folded into the decode split merge == aiter stage2 + the RoPE pass."""

    def test_merge_inv_rope_matches_stage2_then_rope(self):
        from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
            pack_q_fp8_2buff,
        )

        torch.manual_seed(6)
        H, rows = 16, 50_000

        def packed(n, h):
            x = torch.randn(n, h, HEAD, dtype=torch.bfloat16, device=DEV)
            nope = torch.empty(n, h, 512, dtype=torch.float8_e4m3fn, device=DEV)
            rope = torch.empty(n, h, ROPE, dtype=torch.bfloat16, device=DEV)
            pack_q_fp8_2buff(x, nope, rope)
            return nope, rope

        kvn, kvr = packed(rows, 1)
        kvn, kvr = kvn.view(rows, 512), kvr.view(rows, ROPE)
        sink = torch.randn(H, dtype=torch.float32, device=DEV)
        ang = torch.rand(8192, ROPE // 2, device=DEV) * 6.28
        freqs_real = torch.view_as_real(torch.polar(torch.ones_like(ang), ang)).flatten(
            -2
        )
        for T, L, splits in ((1, 640, 8), (8, 640, 8), (5, 300, 4), (3, 128, 1)):
            q, qr = packed(T, H)
            lens = torch.randint(L // 2, L + 1, (T,), device=DEV)
            indptr = torch.zeros(T + 1, dtype=torch.int32, device=DEV)
            indptr[1:] = torch.cumsum(lens, 0)
            idx = torch.randint(
                0, rows, (int(indptr[-1]),), dtype=torch.int32, device=DEV
            )
            pos = torch.randint(0, 8192, (T,), dtype=torch.int64, device=DEV)
            kw = dict(
                q=q,
                q_rope=qr,
                unified_kv=kvn,
                unified_kv_rope=kvr,
                kv_indices=idx,
                kv_indptr=indptr,
                attn_sink=sink,
                v_head_dim=HEAD,
                num_kv_splits=splits,
            )
            ref = runtime.decode_fp8_2buff(**kw).clone()
            runtime.apply_inverse_rope(ref, (freqs_real, pos))
            got = runtime.decode_fp8_2buff(**kw, inv_rope=(freqs_real, pos))
            # the merge loads every split at once and may contract the recurrence
            # into FMAs differently from stage2: a stray 1-ulp bf16 rounding (2e-5
            # of the values at T=8), never more
            diff = (got.float() - ref.float()).abs()
            ulp = ref.float().abs().clamp_min(1e-30) * 2**-7
            self.assertTrue(
                bool((diff <= ulp).all()), (T, L, splits, float((diff / ulp).max()))
            )
            exact = float((got[..., :NOPE] == ref[..., :NOPE]).float().mean())
            self.assertGreaterEqual(exact, 0.9999, (T, L, splits))


@_needs_gfx950
class TestC2VerifyRejection(_Base):
    """Verify parks every even draft row at pos % ring. A rejected tail must not
    clobber the accepted row the next verify pairs with, so verify, reject and
    re-verify leave the pool, the outputs and the ring as a token-by-token decode
    of the accepted tokens does."""

    def test_reverify_after_rejection_matches_decode(self):
        torch.manual_seed(11)
        D = 6  # DSpark block 5 verifies gamma + 1 rows per request
        ring = get_compress_state_ring_size(2, D)
        p0, accepted = [10, 7, 0, 33], [1, 2, 5, 6]
        n_req, row_base = len(p0), 321
        lens = [k + D for k in accepted]
        w = torch.randn(HEAD, dtype=torch.bfloat16, device=DEV)
        fc = freqs(4096)
        state0 = torch.randn(n_req * ring, 2 * HEAD, dtype=torch.float32, device=DEV)
        true_kv = [
            torch.randn(n, 2 * HEAD, dtype=torch.float32, device=DEV) for n in lens
        ]
        bad_kv = [torch.randn(D, 2 * HEAD, dtype=torch.float32, device=DEV) for _ in p0]

        def loc(r, q):  # both halves of a pair share slot loc >> 1; 0 is padding
            return 2 * (1 + 100 * r + q // 2) + q % 2

        def store(kv, state, pools, rows, draft_len):
            n = len(rows)
            out = torch.zeros(n, HEAD, dtype=torch.bfloat16, device=DEV)
            c2_decode_norm_rope_store(
                kv,
                state,
                w,
                torch.tensor([q for _, q in rows], device=DEV),
                torch.tensor([r for r, _ in rows], device=DEV),
                torch.tensor([loc(r, q) for r, q in rows], device=DEV),
                EPS,
                fc,
                pools[0],
                page_size=1,
                ring_size=ring,
                draft_len=draft_len,
                out=out,
                fp8_2buff=True,
                k_cache_rope=pools[1],
                row_base=row_base,
            )
            return out

        def verify(state, pools, starts, kv_rows):
            rows = [(r, s + i) for r, s in enumerate(starts) for i in range(D)]
            return store(torch.cat(kv_rows), state, pools, rows, D)

        pool_rows = row_base + 100 * n_req + 64
        # token by token over the accepted tokens
        ref_state, ref_pools = state0.clone(), two_pool(pool_rows)
        ref_out = [[None] * n for n in lens]
        for i in range(max(lens)):
            live = [r for r in range(n_req) if i < lens[r]]
            kv = torch.cat([true_kv[r][i : i + 1] for r in live])
            out = store(kv, ref_state, ref_pools, [(r, p0[r] + i) for r in live], 1)
            for j, r in enumerate(live):
                ref_out[r][i] = out[j]

        state, pools = state0.clone(), two_pool(pool_rows)
        verify(
            state,
            pools,
            p0,
            [
                torch.cat([t[:k], b[: D - k]])
                for t, b, k in zip(true_kv, bad_kv, accepted)
            ],
        )
        # pairs the rejected tail completed, which the re-verify has to overwrite
        stale = [
            row_base + (loc(r, q) >> 1)
            for r, k in enumerate(accepted)
            for q in range(p0[r] + k, p0[r] + D)
            if q % 2
        ]
        self.assertTrue(stale)
        for row in stale:
            self.assertFalse(torch.equal(pools[1][row], ref_pools[1][row]), row)
        out = verify(
            state,
            pools,
            [p + k for p, k in zip(p0, accepted)],
            [t[k : k + D] for t, k in zip(true_kv, accepted)],
        )
        self.assertTrue(
            torch.equal(pools[0].view(torch.uint8), ref_pools[0].view(torch.uint8))
        )
        self.assertTrue(torch.equal(pools[1], ref_pools[1]))
        for r, k in enumerate(accepted):
            for i in range(D):
                if (p0[r] + k + i) % 2:  # completing rows; parked ones leave out alone
                    self.assertTrue(
                        torch.equal(out[r * D + i], ref_out[r][k + i]), (r, i)
                    )
        # the next step pairs with what the re-verify parked
        nxt = [(r, p0[r] + n) for r, n in enumerate(lens)]
        self.assertTrue(all(q % 2 for _, q in nxt))
        kv = torch.randn(n_req, 2 * HEAD, dtype=torch.float32, device=DEV)
        self.assertTrue(
            torch.equal(
                store(kv, state, pools, nxt, 1),
                store(kv, ref_state, ref_pools, nxt, 1),
            )
        )


class TestDsparkDraftBlockStream(CustomTestCase):
    """The unified draft reads the paged ring draft's block window: every row of a
    block sees its request's last window rows and the whole block. The ring must
    hold window and block at once, or a stream would read one row twice."""

    def test_matches_ring_window(self):
        win = 128
        for block in (5, 3):
            ring = win + block  # get_swa_ring_size under DSpark: window + gamma
            prefixes = [0, 3, win - 1, win, win + 1, 5000]
            bs = len(prefixes)
            slots = torch.randperm(16, device=DEV)[:bs]
            req_rep = slots.repeat_interleave(block)
            positions = torch.tensor(
                prefixes, device=DEV, dtype=torch.int32
            ).repeat_interleave(block) + torch.arange(
                block, device=DEV, dtype=torch.int32
            ).repeat(bs)
            ref, ref_len = BuildDsparkSwaRingIndices.execute(
                seq_lens_casual=positions + 1,
                req_pool_indices_repeated=req_rep,
                valid_from=torch.zeros(16, dtype=torch.int64, device=DEV),
                ring_size=ring,
                block_size=block,
                swa_window=win,
                page_index_aligned_size=64,
            )
            idx, indptr = runtime.build_dspark_block_streams(
                state_slot=req_rep,
                positions=positions,
                block_size=block,
                win=win,
                ring_stride=ring,
            )
            for t in range(bs * block):
                got = idx[int(indptr[t]) : int(indptr[t + 1])].tolist()
                self.assertEqual(got, ref[t, : int(ref_len[t])].tolist(), (block, t))
                self.assertEqual(len(set(got)), len(got), (block, t))


if __name__ == "__main__":
    unittest.main()
