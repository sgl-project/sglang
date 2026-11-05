# SPDX-License-Identifier: Apache-2.0
"""Single-pass top-k(<= 2048) + page transform for DSA decode / MTP-verify rows
(ROCm, <= 128 rows): est (Triton) -> collect (Gluon) -> select (Triton).

  est     per row, sample ~8192 logits (64 probes of 128) and bit-search the
          sample key that leaves ~OVER * topk elements above it in the full row
          -> a per-row threshold (rounded down, so it only adds candidates);
          program 0 also builds the chunk prefix.
  collect ONE pass over the logits on a fixed persistent grid of row-aligned
          chunks; every key >= threshold is appended to the row's candidate
          buffer (CAND entries).
  select  per row, the exact k-th largest key among the candidates, then
          exactly min(L, topk) slots through the page table.  If a row has fewer
          than min(L, topk) candidates or overflows CAND, it selects exactly from
          the whole row instead (slow, rare), so the result is always an exact
          top-k.

Output order is not defined, and when the k-th key has ties the members chosen
among the tied elements depend on candidate arrival order.  The launch shape
depends only on the row count (CUDA-graph safe).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_T = 8192  # logits per collect chunk
_GRID = 2048  # persistent collect programs
_CAND = 8192  # candidates per row
_OVER = 2  # candidate target, x topk
_SMP_CH = 128  # probe length
_SMP_NCH = 64  # probes per row
MAX_ROWS = 128
_CSTR = 64  # int32 gap between rows' counters
_COLLECT_NW = 4
_FALLBACK_BLOCK = 4096


@triton.jit
def _sum_combine(a, b):
    return a + b


@triton.jit
def _okey(b):
    # order-preserving int32 key of a float32 bit pattern
    return tl.where(b >= 0, b, (~b) ^ (-2147483648))


@triton.jit
def _topk_est(
    logits_ptr,
    lengths_ptr,
    pref_ptr,
    ctr_ptr,
    stride_l,
    rows,
    CH: tl.constexpr,
    NCH: tl.constexpr,
    TOPK: tl.constexpr,
    OVER: tl.constexpr,
    T: tl.constexpr,
    RMAX: tl.constexpr,
    CSTR: tl.constexpr,
):
    r = tl.program_id(0)
    # program 0 also builds the chunk prefix (was a separate 5 us dispatch)
    if r == 0:
        ro = tl.arange(0, RMAX)
        msk = ro < rows
        La = tl.load(lengths_ptr + ro, mask=msk, other=0)
        cp = tl.cumsum((La + (T - 1)) // T, 0)
        tl.store(pref_ptr + 1 + ro, cp, mask=msk)
        tl.store(pref_ptr, 0)
    tl.store(ctr_ptr + r * CSTR + 1, 0)
    L = tl.load(lengths_ptr + r)
    k = tl.minimum(L, TOPK)
    span = tl.maximum(CH, L // NCH)
    o = tl.arange(0, CH * NCH)
    p = (o // CH) * span + (o % CH)
    m = p < L
    b = tl.load(logits_ptr + r.to(tl.int64) * stride_l + p, mask=m, other=-1)
    key = _okey(b)
    ns = tl.sum(m.to(tl.int32), 0)
    t = tl.maximum(1, tl.minimum((OVER * k * ns) // L, ns))
    thr = 0
    if tl.sum((key >= 0).to(tl.int32), 0) < t:
        thr = -2147483648
    # bits 30..15 only: rounding the sampled threshold down can only add
    # candidates. Sampling can still overestimate the full-row k-th key;
    # the selector handles that case with its exact underflow fallback.
    for i in tl.static_range(0, 16):
        cand = thr + (1 << (30 - i))
        if tl.sum((key >= cand).to(tl.int32), 0) >= t:
            thr = cand
    tl.store(ctr_ptr + r * CSTR, thr)


@gluon.jit
def _topk_collect(
    logits_ptr,
    lengths_ptr,
    pref_ptr,
    ctr_ptr,
    ckey_ptr,
    cidx_ptr,
    rows,
    stride_l,
    T: gl.constexpr,
    CAND: gl.constexpr,
    RMAX: gl.constexpr,
    GRID: gl.constexpr,
    CSTR: gl.constexpr,
    NW: gl.constexpr,
):
    # Thread-local compaction with a pinned layout: thread t of the CTA owns the R
    # 16-byte vectors (a, t, 0..3) of the chunk. The slot of a selected element is
    # base + (count of the threads before t) + (its rank inside thread t): per-thread
    # sums and in-thread ranks are register ops, the NT-wide scan (one value per
    # thread) is the only cross-thread step, and the scattered stores keep the load
    # layout (no LDS round trip). Within-chunk order is a permutation -- free, since
    # the selector's output order is unconstrained and cross-chunk order was already
    # atomic arrival order.
    NT: gl.constexpr = 64 * NW
    R: gl.constexpr = T // (NT * 4)
    L3: gl.constexpr = gl.BlockedLayout([R, 1, 4], [1, 64, 1], [1, NW, 1], [2, 1, 0])
    LRT: gl.constexpr = gl.SliceLayout(2, L3)
    L2: gl.constexpr = gl.BlockedLayout([R, 1], [1, 64], [1, NW], [1, 0])
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    pid = gl.program_id(0)
    ro = gl.arange(0, RMAX, layout=L1)
    pref = gl.load(pref_ptr + 1 + ro, mask=ro < rows, other=2147483647)
    total = gl.load(pref_ptr + rows)
    a = gl.arange(0, R, layout=gl.SliceLayout(1, LRT))
    t = gl.arange(0, NT, layout=gl.SliceLayout(0, LRT))
    v = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    va = gl.expand_dims(gl.expand_dims(a, 1) * (NT * 4) + gl.expand_dims(t, 0) * 4, 2)
    loc = va + gl.expand_dims(gl.expand_dims(v, 0), 1)
    for c in range(pid, total, GRID):
        r = gl.sum((c >= pref).to(gl.int32), 0)
        start = (c - gl.load(pref_ptr + r)) * T
        L = gl.load(lengths_ptr + r)
        thr = gl.load(ctr_ptr + r * CSTR)
        idx = start + loc
        # 16-byte-granular load mask (rows are 4-aligned and >= L wide, so the tail
        # vector's extra elements are in-bounds); the exact `idx < L` gates selection.
        # Buffer base = the kernel argument (uniform), row folded into the 32-bit offset
        # (host asserts the logits fit in 2**31 bytes): a base derived from the reduced
        # `r` is not provably uniform and every buffer op becomes a waterfall loop.
        b = gl.amd.cdna3.buffer_load(
            ptr=logits_ptr, offsets=r * stride_l + idx, mask=(start + va) < L, other=-1
        )
        key = gl.where(b >= 0, b, (~b) ^ (-2147483648))
        si = ((idx < L) & (key >= thr)).to(gl.int32)
        s_v = gl.sum(si, 2)
        cnt = gl.sum(s_v, 0)
        n = gl.sum(cnt, 0)
        if n > 0:
            base = gl.atomic_add(ctr_ptr + r * CSTR + 1, n, sem="relaxed")
            # tt.scan only lowers on blocked layouts: L1 / L2 are the same thread maps
            # as the slice layouts (no data movement).
            cnt_b = gl.convert_layout(cnt, L1)
            tb = gl.convert_layout(
                gl.associative_scan(cnt_b, 0, _sum_combine) - cnt_b,
                gl.SliceLayout(0, LRT),
            )
            s_vb = gl.convert_layout(s_v, L2)
            rv = gl.convert_layout(
                gl.associative_scan(s_vb, 0, _sum_combine) - s_vb, LRT
            )
            rk = gl.expand_dims(rv, 2) + (gl.associative_scan(si, 2, _sum_combine) - si)
            p = base + gl.expand_dims(gl.expand_dims(tb, 0), 2) + rk
            mm = (si != 0) & (p < CAND)
            gl.amd.cdna3.buffer_store(key, ptr=ckey_ptr, offsets=r * CAND + p, mask=mm)
            gl.amd.cdna3.buffer_store(idx, ptr=cidx_ptr, offsets=r * CAND + p, mask=mm)


@triton.jit
def _row_threshold(row_ptr, L, k, BLOCK: tl.constexpr):
    """Exact k-th largest key of a whole row (31-step bit search over the row)."""
    thr = 0
    cnt0 = 0
    for s in range(0, L, BLOCK):
        o = s + tl.arange(0, BLOCK)
        b = tl.load(row_ptr + o, mask=o < L, other=-1)
        cnt0 += tl.sum(((o < L) & (_okey(b) >= 0)).to(tl.int32), 0)
    if cnt0 < k:
        thr = -2147483648
    for i in tl.static_range(0, 31):
        cand = thr + (1 << (30 - i))
        c = 0
        for s in range(0, L, BLOCK):
            o = s + tl.arange(0, BLOCK)
            b = tl.load(row_ptr + o, mask=o < L, other=-1)
            c += tl.sum(((o < L) & (_okey(b) >= cand)).to(tl.int32), 0)
        if c >= k:
            thr = cand
    return thr


@triton.jit
def _topk_select(
    ckey_ptr,
    cidx_ptr,
    ctr_ptr,
    lengths_ptr,
    logits_ptr,
    stride_l,
    page_ptr,
    stride_p,
    out_ptr,
    CAND: tl.constexpr,
    TOPK: tl.constexpr,
    CSTR: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    n_all = tl.load(ctr_ptr + r * CSTR + 1)
    L = tl.load(lengths_ptr + r)
    k = tl.minimum(L, TOPK)
    pt = page_ptr + r.to(tl.int64) * stride_p
    out_cols = tl.arange(0, TOPK)
    if (n_all < k) | (n_all > CAND):
        # The sampled threshold was too high (underflow) or too low (overflow):
        # select exactly from the whole row instead. Rare and slow.
        row = logits_ptr + r.to(tl.int64) * stride_l
        thr = _row_threshold(row, L, k, BLOCK)
        base = 0
        above = 0
        for s in range(0, L, BLOCK):
            o = s + tl.arange(0, BLOCK)
            key = _okey(tl.load(row + o, mask=o < L, other=-1))
            above += tl.sum(((o < L) & (key > thr)).to(tl.int32), 0)
        room = k - above
        eq_seen = 0
        for s in range(0, L, BLOCK):
            o = s + tl.arange(0, BLOCK)
            key = _okey(tl.load(row + o, mask=o < L, other=-1))
            gt = (o < L) & (key > thr)
            eq = (o < L) & (key == thr)
            eq_rank = eq_seen + tl.cumsum(eq.to(tl.int32), 0) - eq.to(tl.int32)
            take = gt | (eq & (eq_rank < room))
            ti = take.to(tl.int32)
            pos = base + tl.cumsum(ti, 0) - ti
            slot = (
                tl.load(pt + (o >> PAGE_SHIFT), mask=take, other=0) << PAGE_SHIFT
            ) + (o & ((1 << PAGE_SHIFT) - 1))
            tl.store(out_ptr + r * TOPK + pos, slot, mask=take)
            base += tl.sum(ti, 0)
            eq_seen += tl.sum(eq.to(tl.int32), 0)
        tl.store(
            out_ptr + r * TOPK + out_cols,
            tl.full((TOPK,), -1, tl.int32),
            mask=out_cols >= base,
        )
    else:
        n = n_all
        o = tl.arange(0, CAND)
        m = o < n
        cb = r.to(tl.int64) * CAND
        key = tl.load(ckey_ptr + cb + o, mask=m, other=-2147483648)
        thr = 0
        if tl.sum((key >= 0).to(tl.int32), 0) < k:
            thr = -2147483648
        for i in tl.static_range(0, 31):
            cand = thr + (1 << (30 - i))
            if tl.sum((key >= cand).to(tl.int32), 0) >= k:
                thr = cand
        gt = m & (key > thr)
        gti = gt.to(tl.int32)
        eq = m & (key == thr)
        eqi = eq.to(tl.int32)
        # one packed CAND-wide scan carries both ranks (each field < CAND <= 65536)
        cs = tl.cumsum(gti + (eqi * 65536), 0)
        rank_gt = (cs & 65535) - gti
        rank_eq = (cs // 65536) - eqi
        above = tl.sum(gti, 0)
        room = k - above
        take = gt | (eq & (rank_eq < room))
        n_take = tl.sum(take.to(tl.int32), 0)
        pos = rank_gt + tl.minimum(rank_eq, room)
        idx = tl.load(cidx_ptr + cb + o, mask=take, other=0)
        slot = (tl.load(pt + (idx >> PAGE_SHIFT), mask=take, other=0) << PAGE_SHIFT) + (
            idx & ((1 << PAGE_SHIFT) - 1)
        )
        tl.store(
            out_ptr + r * TOPK + out_cols,
            tl.full((TOPK,), -1, tl.int32),
            mask=out_cols >= n_take,
        )
        tl.store(out_ptr + r * TOPK + pos, slot, mask=take)


def topk_transform_paged_triton(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    topk: int,
) -> torch.Tensor:
    """Top-``topk`` of ``logits[row, :lengths[row]]`` mapped to physical KV slots
    ``page_table[row, p // page_size] * page_size + p % page_size``; ``-1`` padded.

    ``logits`` fp32 ``[rows, >= max(lengths)]`` with unit column stride,
    ``page_table`` int32 ``[rows, pages]``.  ``rows <= MAX_ROWS`` and
    ``rows * logits.stride(0) * 4 < 2**31`` (32-bit buffer offsets).
    """
    rows = logits.shape[0]
    dev = logits.device
    assert 0 < rows <= MAX_ROWS, rows
    assert page_size & (page_size - 1) == 0, page_size
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    out = torch.empty((rows, topk), dtype=torch.int32, device=dev)
    ctr = torch.empty((rows, _CSTR), dtype=torch.int32, device=dev)
    pref = torch.empty((rows + 1,), dtype=torch.int32, device=dev)
    ckey = torch.empty((rows, _CAND), dtype=torch.int32, device=dev)
    cidx = torch.empty((rows, _CAND), dtype=torch.int32, device=dev)
    li = logits.view(torch.int32)
    sl = li.stride(0)
    assert rows * sl * 4 < 2**31 and rows * _CAND * 4 < 2**31
    _topk_est[(rows,)](
        li,
        lengths,
        pref,
        ctr,
        sl,
        rows,
        CH=_SMP_CH,
        NCH=_SMP_NCH,
        TOPK=topk,
        OVER=_OVER,
        T=_T,
        RMAX=MAX_ROWS,
        CSTR=_CSTR,
        num_warps=8,
    )
    _topk_collect[(_GRID,)](
        li,
        lengths,
        pref,
        ctr,
        ckey,
        cidx,
        rows,
        sl,
        T=_T,
        CAND=_CAND,
        RMAX=MAX_ROWS,
        GRID=_GRID,
        CSTR=_CSTR,
        NW=_COLLECT_NW,
        num_warps=_COLLECT_NW,
    )
    _topk_select[(rows,)](
        ckey,
        cidx,
        ctr,
        lengths,
        li,
        sl,
        page_table,
        page_table.stride(0),
        out,
        CAND=_CAND,
        TOPK=topk,
        CSTR=_CSTR,
        PAGE_SHIFT=page_size.bit_length() - 1,
        BLOCK=_FALLBACK_BLOCK,
        num_warps=8,
    )
    return out
