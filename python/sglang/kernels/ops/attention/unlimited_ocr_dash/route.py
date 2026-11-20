"""Fused Triton DASH routing: page summaries, remote block scores, and
1.5-entmax selection with centered log-prior bias, capped by a hard top-k budget.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import torch
import triton
import triton.language as tl

# The tau bisection reaches the fp32 precision limit after 24 iterations.
ENTMAX_ITERS = 24
# The guarded budget-cap bisection retains its full production precision.
ENTMAX_ITERS_CAP = 40
# Decode routing uses a fixed 128-column, window-relative page scan.
SCAN_TILE = 128
# Score columns per program; a scan tile must be a whole number of them.
_SCORE_CHUNK = 64


def _want_tiled(bw_full: int) -> bool:
    """Use the tiled router when the full-width companion reaches 8192 columns."""
    return bw_full >= 8192


# Same finite sentinels as the torch path / training.
SCORE_NEG = -1.0e9
ENTMAX_NEG = -1.0e4
ACT_CLAMP = 1.0e4
# Triton kernels may only read module globals created as `tl.constexpr(...)`
# (the annotated form `x: tl.constexpr = v` is NOT accepted).
_SCORE_NEG = tl.constexpr(SCORE_NEG)
_ENTMAX_NEG = tl.constexpr(ENTMAX_NEG)
_ACT_CLAMP = tl.constexpr(ACT_CLAMP)
_ENTMAX_ITERS = tl.constexpr(ENTMAX_ITERS)
_ENTMAX_ITERS_CAP = tl.constexpr(ENTMAX_ITERS_CAP)


@triton.jit
def _kbar_page_kernel(
    kv_ptr,
    hc_ptr,
    tok_ptr,
    page_ptr,
    out_ptr,
    stride_kv_p,
    stride_kv_t,
    stride_kv_d,
    stride_out_p,
    inv_sqrt_d,
    D: tl.constexpr,
    BS: tl.constexpr,
):
    """head_cls summary of one KV page -> out[page] (fp32 [D])."""
    pid = tl.program_id(0)
    page = tl.load(page_ptr + pid).to(tl.int64)
    ntok = tl.load(tok_ptr + pid)

    off_t = tl.arange(0, BS)
    off_d = tl.arange(0, D)
    valid = off_t < ntok
    keys = tl.load(
        kv_ptr
        + page * stride_kv_p
        + off_t[:, None] * stride_kv_t
        + off_d[None, :] * stride_kv_d,
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)  # [BS, D]
    hc = tl.load(hc_ptr + off_d).to(tl.float32)  # [D]

    aw = tl.sum(keys * hc[None, :], axis=1) * inv_sqrt_d  # [BS]
    aw = tl.minimum(tl.maximum(aw, -_ACT_CLAMP), _ACT_CLAMP)
    # NaN-safe masked softmax: row max over VALID tokens only (matches training).
    neg = float("-inf")
    row_max = tl.max(tl.where(valid, aw, neg), axis=0)
    row_max = tl.where(ntok > 0, row_max, 0.0)
    e = tl.where(valid, tl.exp(aw - row_max), 0.0)
    denom = tl.maximum(tl.sum(e, axis=0), 1e-6)
    w = e / denom
    k_bar = tl.sum(keys * w[:, None], axis=0)  # [D]
    k_bar = tl.where(k_bar == k_bar, k_bar, 0.0)  # nan -> 0
    # ntok == 0 marks a non-page slot (dense [R, W] prefill plan / zero-length
    # CG padding rows); storing would clobber another request's page.
    if ntok > 0:
        tl.store(out_ptr + page * stride_out_p + off_d, k_bar)


@triton.jit
def _score_kernel(
    q_ptr,
    kbar_ptr,
    bt_ptr,
    sl_ptr,
    out_ptr,
    lo_ptr,
    fit_ptr,
    stride_q_n,
    stride_kbar_p,
    stride_bt_r,
    stride_out_n,
    W,
    inv_sqrt_d,
    block_size,
    local_blocks,
    dql,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    REL: tl.constexpr,
    FIT_SEL: tl.constexpr,
):
    """Block scores for one query row and one chunk of blocks.

    ``REL`` stores block ``lo[row] + j`` at ``out[row, j]``; the image mask is left
    to ``_entmax_kernel``, whose per-query fallback needs every remote score.
    """
    row = tl.program_id(0)
    if FIT_SEL == 1:
        if tl.load(fit_ptr + row) == 0:
            return
    if FIT_SEL == 2:
        if tl.load(fit_ptr + row) != 0:
            return
    c0 = tl.program_id(1) * CHUNK
    r = row // dql

    seq = tl.load(sl_ptr + r).to(tl.int32)
    q_pos = seq - dql + (row % dql)
    q_blk = q_pos // block_size

    off_j = c0 + tl.arange(0, CHUNK)  # storage column
    base = 0
    if REL:
        base = tl.load(lo_ptr + row).to(tl.int32)
    off_w = base + off_j  # absolute block id
    in_w = off_w < W
    # A chunk beyond the causal or sequence bound can only contain sentinels.
    if base + c0 > q_blk - local_blocks or (base + c0) * block_size >= seq:
        tl.store(
            out_ptr + row * stride_out_n + off_j,
            tl.full((CHUNK,), _SCORE_NEG, tl.float32),
            mask=in_w,
        )
        return
    keep = in_w & (off_w <= (q_blk - local_blocks)) & (off_w * block_size < seq)

    off_d = tl.arange(0, D)
    pages = tl.load(bt_ptr + r * stride_bt_r + off_w, mask=in_w, other=0).to(tl.int64)
    kb = tl.load(
        kbar_ptr + pages[:, None] * stride_kbar_p + off_d[None, :],
        mask=keep[:, None],
        other=0.0,
    ).to(tl.float32)  # [CHUNK, D]
    q = tl.load(q_ptr + row * stride_q_n + off_d).to(tl.float32)  # [D]
    z = tl.sum(kb * q[None, :], axis=1) * inv_sqrt_d
    z = tl.minimum(tl.maximum(z, -_ACT_CLAMP), _ACT_CLAMP)
    tl.store(
        out_ptr + row * stride_out_n + off_j, tl.where(keep, z, _SCORE_NEG), mask=in_w
    )


@triton.jit
def _entmax_kernel(
    sc_ptr,
    img_ptr,
    base_ptr,
    fit_ptr,
    out_idx_ptr,
    out_bias_ptr,
    stride_sc_n,
    stride_img_n,
    stride_oi_n,
    stride_ob_n,
    W,
    idx_scale,
    prior_floor,
    inv_sigma,
    K,
    M: tl.constexpr,
    BLOCK_W: tl.constexpr,
    HAS_IMG: tl.constexpr,
    HAS_WIN: tl.constexpr,
    FIT_SEL: tl.constexpr,
    REL: tl.constexpr = False,
):
    """Route one query row with sort-free 1.5-entmax.

    ``HAS_WIN`` shifts to ``base[row]``; ``REL`` makes score columns relative to
    that base. ``FIT_SEL`` partitions rows across narrow and full-width launches.
    """
    row = tl.program_id(0)
    if FIT_SEL == 1:
        if tl.load(fit_ptr + row) == 0:
            return
    if FIT_SEL == 2:
        if tl.load(fit_ptr + row) != 0:
            return
    base = 0
    if HAS_WIN:
        base = tl.load(base_ptr + row).to(tl.int32)
    off_w = base + tl.arange(0, BLOCK_W)
    in_w = off_w < W
    off_s = off_w
    if REL:
        off_s = tl.arange(0, BLOCK_W)

    s = tl.load(sc_ptr + row * stride_sc_n + off_s, mask=in_w, other=_SCORE_NEG)
    remote = s > (_SCORE_NEG * 0.5)
    cand = remote
    if HAS_IMG:
        img = tl.load(img_ptr + row * stride_img_n + off_w, mask=in_w, other=0).to(
            tl.int1
        )
        cand_img = remote & img
        # per-query fallback: the image mask left nothing -> drop the constraint
        cand = tl.where(tl.sum(cand_img.to(tl.int32), axis=0) > 0, cand_img, remote)
    has_any = tl.sum(cand.to(tl.int32), axis=0) > 0

    x = tl.where(cand, s * idx_scale, _ENTMAX_NEG)
    x = tl.where(x == x, x, _ENTMAX_NEG)  # nan -> sentinel
    xm = tl.max(tl.where(cand, x, float("-inf")), axis=0)
    xm = tl.where(has_any, xm, 0.0)
    z = (x - xm) * 0.5

    # Solve the entmax threshold over [-1, 0]. fp64 reductions make eager and
    # CUDA-graph tile widths round identically.
    lo = -1.0
    hi = 0.0
    for _ in range(_ENTMAX_ITERS):
        mid = (lo + hi) * 0.5
        p = tl.maximum(z - mid, 0.0)
        ssum = tl.sum(tl.where(cand, p * p, 0.0).to(tl.float64), axis=0).to(tl.float32)
        lo = tl.where(ssum > 1.0, mid, lo)
        hi = tl.where(ssum > 1.0, hi, mid)
    tau = (lo + hi) * 0.5

    prior = tl.maximum(z - tau, 0.0)
    prior = prior * prior

    # Hard top-K includes candidates whose entmax prior is exactly zero.
    zmin = tl.min(tl.where(cand, z, float("inf")), axis=0)
    lo2 = tl.where(has_any, zmin - 1.0, -1.0)
    hi2 = 0.0
    active = cand
    if tl.sum(cand.to(tl.int32), axis=0) > K:
        for _ in range(_ENTMAX_ITERS_CAP):
            mid = (lo2 + hi2) * 0.5
            cnt = tl.sum((cand & (z > mid)).to(tl.int32), axis=0)
            lo2 = tl.where(cnt > K, mid, lo2)
            hi2 = tl.where(cnt > K, hi2, mid)
        active = cand & (z > hi2)

    logp = tl.log(tl.maximum(prior, prior_floor))
    cnt_a = tl.maximum(tl.sum(active.to(tl.int32), axis=0), 1)
    # fp64 for the same tree-order reason as the tau bisection: mu feeds every
    # emitted bias, so its 1-ulp width dependence moves the output text.
    mu = tl.sum(tl.where(active, logp, 0.0).to(tl.float64), axis=0).to(
        tl.float32
    ) / cnt_a.to(tl.float32)
    bias = tl.where(active, (logp - mu) * inv_sigma, 0.0)

    # pack the active blocks at the front; -1 / 0.0 for the unused slots
    off_m = tl.arange(0, M)
    tl.store(out_idx_ptr + row * stride_oi_n + off_m, tl.full((M,), -1, tl.int32))
    tl.store(out_bias_ptr + row * stride_ob_n + off_m, tl.zeros((M,), tl.float32))
    pos = tl.cumsum(active.to(tl.int32), axis=0) - 1
    ok = active & (pos < M)
    tl.store(out_idx_ptr + row * stride_oi_n + pos, off_w.to(tl.int32), mask=ok)
    tl.store(out_bias_ptr + row * stride_ob_n + pos, bias, mask=ok)


@triton.jit
def _score_tiled_kernel(
    q_ptr,
    kbar_ptr,
    bt_ptr,
    sl_ptr,
    out_ptr,
    lo_ptr,
    hi_ptr,
    stride_q_n,
    stride_kbar_p,
    stride_bt_r,
    stride_out_n,
    W,
    inv_sqrt_d,
    block_size,
    local_blocks,
    dql,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    TILE: tl.constexpr,
):
    """Compute window-relative decode scores; the grid visits only the live span."""
    row = tl.program_id(0)
    cj = tl.program_id(1) * CHUNK
    r = row // dql

    seq = tl.load(sl_ptr + r).to(tl.int32)
    q_pos = seq - dql + (row % dql)
    q_blk = q_pos // block_size
    lo = tl.load(lo_ptr + row).to(tl.int32)
    hi = tl.load(hi_ptr + row).to(tl.int32)
    span = tl.maximum(hi - lo, 0)
    n_tiles = (span + TILE - 1) // TILE

    off_d = tl.arange(0, D)
    q = tl.load(q_ptr + row * stride_q_n + off_d).to(tl.float32)  # [D]
    for t in range(n_tiles):
        off_j = t * TILE + cj + tl.arange(0, CHUNK)  # window-relative column
        off_w = lo + off_j  # absolute block id
        in_w = (off_j < span) & (off_w < W)
        keep = in_w & (off_w <= (q_blk - local_blocks)) & (off_w * block_size < seq)
        pages = tl.load(bt_ptr + r * stride_bt_r + off_w, mask=keep, other=0).to(
            tl.int64
        )
        kb = tl.load(
            kbar_ptr + pages[:, None] * stride_kbar_p + off_d[None, :],
            mask=keep[:, None],
            other=0.0,
        ).to(tl.float32)  # [CHUNK, D]
        z = tl.sum(kb * q[None, :], axis=1) * inv_sqrt_d
        z = tl.minimum(tl.maximum(z, -_ACT_CLAMP), _ACT_CLAMP)
        tl.store(
            out_ptr + row * stride_out_n + off_j,
            tl.where(keep, z, _SCORE_NEG),
            mask=in_w,
        )


@triton.jit
def _tile_load(
    sc_ptr,
    img_ptr,
    row,
    stride_sc_n,
    stride_img_n,
    lo,
    t,
    span,
    W,
    use_img,
    idx_scale,
    HAS_IMG: tl.constexpr,
    TILE: tl.constexpr,
):
    """Load one overflow tile for a multi-tile row."""
    off_j = t * TILE + tl.arange(0, TILE)
    off_w = lo + off_j
    in_w = (off_j < span) & (off_w < W)
    s = tl.load(sc_ptr + row * stride_sc_n + off_j, mask=in_w, other=_SCORE_NEG)
    cand = s > (_SCORE_NEG * 0.5)
    if HAS_IMG:
        img = tl.load(img_ptr + row * stride_img_n + off_w, mask=in_w, other=0).to(
            tl.int1
        )
        cand = tl.where(use_img, cand & img, cand)
    x = tl.where(cand, s * idx_scale, _ENTMAX_NEG)
    x = tl.where(x == x, x, _ENTMAX_NEG)  # nan -> sentinel
    return off_w, s, cand, x


@triton.jit
def _entmax_tiled_kernel(
    sc_ptr,
    img_ptr,
    lo_ptr,
    hi_ptr,
    out_idx_ptr,
    out_bias_ptr,
    stride_sc_n,
    stride_img_n,
    stride_oi_n,
    stride_ob_n,
    W,
    idx_scale,
    prior_floor,
    inv_sigma,
    K,
    M: tl.constexpr,
    TILE: tl.constexpr,
    HAS_IMG: tl.constexpr,
):
    """Run tiled entmax for every decode row.

    The first tile remains register-resident; wider rows reload overflow tiles.
    """
    row = tl.program_id(0)
    lo = tl.load(lo_ptr + row).to(tl.int32)
    hi = tl.load(hi_ptr + row).to(tl.int32)
    span = tl.maximum(hi - lo, 0)
    n_tiles = (span + TILE - 1) // TILE
    # -1 / 0.0 for every slot first: a row with no candidate writes nothing below.
    off_m = tl.arange(0, M)
    tl.store(out_idx_ptr + row * stride_oi_n + off_m, tl.full((M,), -1, tl.int32))
    tl.store(out_bias_ptr + row * stride_ob_n + off_m, tl.zeros((M,), tl.float32))

    # The scalar branch keeps the one-tile fast body and multi-tile body merged.
    if n_tiles > 1:
        _entmax_tiled_body(
            sc_ptr,
            img_ptr,
            out_idx_ptr,
            out_bias_ptr,
            row,
            lo,
            span,
            n_tiles,
            stride_sc_n,
            stride_img_n,
            stride_oi_n,
            stride_ob_n,
            W,
            idx_scale,
            prior_floor,
            inv_sigma,
            K,
            M=M,
            TILE=TILE,
            HAS_IMG=HAS_IMG,
            MULTI=True,
        )
    else:
        _entmax_tiled_body(
            sc_ptr,
            img_ptr,
            out_idx_ptr,
            out_bias_ptr,
            row,
            lo,
            span,
            n_tiles,
            stride_sc_n,
            stride_img_n,
            stride_oi_n,
            stride_ob_n,
            W,
            idx_scale,
            prior_floor,
            inv_sigma,
            K,
            M=M,
            TILE=TILE,
            HAS_IMG=HAS_IMG,
            MULTI=False,
        )


@triton.jit
def _entmax_tiled_body(
    sc_ptr,
    img_ptr,
    out_idx_ptr,
    out_bias_ptr,
    row,
    lo,
    span,
    n_tiles,
    stride_sc_n,
    stride_img_n,
    stride_oi_n,
    stride_ob_n,
    W,
    idx_scale,
    prior_floor,
    inv_sigma,
    K,
    M: tl.constexpr,
    TILE: tl.constexpr,
    HAS_IMG: tl.constexpr,
    MULTI: tl.constexpr,
):
    """Run the tiled entmax passes for one row."""
    if not MULTI:
        n_tiles = 1
    # ---- tile 0: loaded once, register-resident for the rest of the kernel ----
    off_t = tl.arange(0, TILE)
    off_w0 = lo + off_t
    in0 = (off_t < span) & (off_w0 < W)
    s0 = tl.load(sc_ptr + row * stride_sc_n + off_t, mask=in0, other=_SCORE_NEG)
    rem0 = s0 > (_SCORE_NEG * 0.5)
    img0 = rem0
    if HAS_IMG:
        img0 = tl.load(img_ptr + row * stride_img_n + off_w0, mask=in0, other=0).to(
            tl.int1
        )

    # P0: per-query image fallback -- does `remote & img` leave anything at all?
    use_img = False
    if HAS_IMG:
        n_img = tl.sum((rem0 & img0).to(tl.int32), axis=0)
        if n_tiles > 1:
            for t in range(1, n_tiles):
                off_j = t * TILE + off_t
                off_w = lo + off_j
                in_w = (off_j < span) & (off_w < W)
                s = tl.load(
                    sc_ptr + row * stride_sc_n + off_j, mask=in_w, other=_SCORE_NEG
                )
                im = tl.load(
                    img_ptr + row * stride_img_n + off_w, mask=in_w, other=0
                ).to(tl.int1)
                n_img += tl.sum(((s > (_SCORE_NEG * 0.5)) & im).to(tl.int32), axis=0)
        use_img = n_img > 0

    cand0 = rem0
    if HAS_IMG:
        cand0 = tl.where(use_img, rem0 & img0, rem0)
    x0 = tl.where(cand0, s0 * idx_scale, _ENTMAX_NEG)
    x0 = tl.where(x0 == x0, x0, _ENTMAX_NEG)  # nan -> sentinel

    # P1: max / min of x over the candidate set, and the candidate count.
    xm = tl.max(tl.where(cand0, x0, -float("inf")), axis=0)
    xmin = tl.min(tl.where(cand0, x0, float("inf")), axis=0)
    n_cand = tl.sum(cand0.to(tl.int32), axis=0)
    if n_tiles > 1:
        for t in range(1, n_tiles):
            _ow, _s, cand, x = _tile_load(
                sc_ptr,
                img_ptr,
                row,
                stride_sc_n,
                stride_img_n,
                lo,
                t,
                span,
                W,
                use_img,
                idx_scale,
                HAS_IMG=HAS_IMG,
                TILE=TILE,
            )
            xm = tl.maximum(xm, tl.max(tl.where(cand, x, -float("inf")), axis=0))
            xmin = tl.minimum(xmin, tl.min(tl.where(cand, x, float("inf")), axis=0))
            n_cand += tl.sum(cand.to(tl.int32), axis=0)
    has_any = n_cand > 0
    xm = tl.where(has_any, xm, 0.0)
    z0 = (x0 - xm) * 0.5
    zmin = (xmin - xm) * 0.5

    # P2: tau* solves sum(clamp(z - tau, 0)^2) = 1 and lies in [max(z) - 1,
    # max(z)] = [-1, 0] after the shift. Monotone decreasing -> bisection.
    blo = -1.0
    bhi = 0.0
    for _ in range(_ENTMAX_ITERS):
        mid = (blo + bhi) * 0.5
        p = tl.maximum(z0 - mid, 0.0)
        ssum = tl.sum(tl.where(cand0, p * p, 0.0), axis=0)
        if n_tiles > 1:
            for t in range(1, n_tiles):
                _ow, _s, cand, x = _tile_load(
                    sc_ptr,
                    img_ptr,
                    row,
                    stride_sc_n,
                    stride_img_n,
                    lo,
                    t,
                    span,
                    W,
                    use_img,
                    idx_scale,
                    HAS_IMG=HAS_IMG,
                    TILE=TILE,
                )
                pt = tl.maximum((x - xm) * 0.5 - mid, 0.0)
                ssum += tl.sum(tl.where(cand, pt * pt, 0.0), axis=0)
        blo = tl.where(ssum > 1.0, mid, blo)
        bhi = tl.where(ssum > 1.0, bhi, mid)
    tau = (blo + bhi) * 0.5

    # P3: hard top-K threshold, skipped when every candidate already fits.
    lo2 = tl.where(has_any, zmin - 1.0, -1.0)
    hi2 = 0.0
    do_cap = n_cand > K
    if do_cap:
        for _ in range(_ENTMAX_ITERS_CAP):
            mid = (lo2 + hi2) * 0.5
            cnt = tl.sum((cand0 & (z0 > mid)).to(tl.int32), axis=0)
            if n_tiles > 1:
                for t in range(1, n_tiles):
                    _ow, _s, cand, x = _tile_load(
                        sc_ptr,
                        img_ptr,
                        row,
                        stride_sc_n,
                        stride_img_n,
                        lo,
                        t,
                        span,
                        W,
                        use_img,
                        idx_scale,
                        HAS_IMG=HAS_IMG,
                        TILE=TILE,
                    )
                    cnt += tl.sum((cand & ((x - xm) * 0.5 > mid)).to(tl.int32), axis=0)
            lo2 = tl.where(cnt > K, mid, lo2)
            hi2 = tl.where(cnt > K, hi2, mid)
    zthr = tl.where(do_cap, hi2, -float("inf"))

    # P4: mu is the mean of log(prior) over the selected slots.
    prior0 = tl.maximum(z0 - tau, 0.0)
    prior0 = prior0 * prior0
    act0 = cand0 & (z0 > zthr)
    logp0 = tl.log(tl.maximum(prior0, prior_floor))
    sum_logp = tl.sum(tl.where(act0, logp0, 0.0), axis=0)
    n_sel = tl.sum(act0.to(tl.int32), axis=0)
    if n_tiles > 1:
        for t in range(1, n_tiles):
            _ow, _s, cand, x = _tile_load(
                sc_ptr,
                img_ptr,
                row,
                stride_sc_n,
                stride_img_n,
                lo,
                t,
                span,
                W,
                use_img,
                idx_scale,
                HAS_IMG=HAS_IMG,
                TILE=TILE,
            )
            z = (x - xm) * 0.5
            pr = tl.maximum(z - tau, 0.0)
            pr = pr * pr
            act = cand & (z > zthr)
            lp = tl.log(tl.maximum(pr, prior_floor))
            sum_logp += tl.sum(tl.where(act, lp, 0.0), axis=0)
            n_sel += tl.sum(act.to(tl.int32), axis=0)
    mu = sum_logp / tl.maximum(n_sel, 1).to(tl.float32)

    # P5: pack the active blocks at the front, carrying the slot offset across
    # tiles (`n_done`) so the packing keeps the ascending-column order.
    bias0 = tl.where(act0, (logp0 - mu) * inv_sigma, 0.0)
    pos0 = tl.cumsum(act0.to(tl.int32), axis=0) - 1
    ok0 = act0 & (pos0 < M)
    tl.store(out_idx_ptr + row * stride_oi_n + pos0, off_w0.to(tl.int32), mask=ok0)
    tl.store(out_bias_ptr + row * stride_ob_n + pos0, bias0, mask=ok0)
    if n_tiles > 1:
        n_done = tl.sum(act0.to(tl.int32), axis=0)
        for t in range(1, n_tiles):
            off_w, _s, cand, x = _tile_load(
                sc_ptr,
                img_ptr,
                row,
                stride_sc_n,
                stride_img_n,
                lo,
                t,
                span,
                W,
                use_img,
                idx_scale,
                HAS_IMG=HAS_IMG,
                TILE=TILE,
            )
            z = (x - xm) * 0.5
            pr = tl.maximum(z - tau, 0.0)
            pr = pr * pr
            act = cand & (z > zthr)
            lp = tl.log(tl.maximum(pr, prior_floor))
            bias = tl.where(act, (lp - mu) * inv_sigma, 0.0)
            pos = tl.cumsum(act.to(tl.int32), axis=0) - 1 + n_done
            ok = act & (pos < M)
            tl.store(out_idx_ptr + row * stride_oi_n + pos, off_w.to(tl.int32), mask=ok)
            tl.store(out_bias_ptr + row * stride_ob_n + pos, bias, mask=ok)
            n_done += tl.sum(act.to(tl.int32), axis=0)


@triton.jit
def _win_bounds_kernel(
    allow_ptr,
    sl_ptr,
    lo_ptr,
    hi_ptr,
    fit_ptr,
    stride_a_n,
    W,
    WQ,
    block_size,
    local_blocks,
    dql,
    tile,
    CHUNK: tl.constexpr,
):
    """Compute each decode query's remote page span.

    Allowed live blocks before the local window define ``(lo, hi)``. Empty page
    spans fall back to the live remote width; ``fit`` selects the narrow tile.
    """
    row = tl.program_id(0)
    r = row // dql
    seq = tl.load(sl_ptr + r).to(tl.int32)
    q_pos = seq - dql + (row % dql)
    q_blk = q_pos // block_size

    lo = W
    hi = -1
    for c0 in range(0, W, CHUNK):
        off = c0 + tl.arange(0, CHUNK)
        m = (off < W) & (off < WQ)
        a = tl.load(allow_ptr + row * stride_a_n + off, mask=m, other=0)
        keep = (a != 0) & m & (off <= (q_blk - local_blocks)) & (off * block_size < seq)
        lo = tl.minimum(lo, tl.min(tl.where(keep, off, W), axis=0))
        hi = tl.maximum(hi, tl.max(tl.where(keep, off, -1), axis=0))

    any_r = hi >= 0
    # Fall back to the sequence's live remote width, not graph-static `W`.
    # Rows without remote blocks use an empty span and stay on the narrow path.
    n_blk = (seq + block_size - 1) // block_size
    w_seq = tl.minimum(W, tl.maximum(n_blk, 1))
    no_remote = (seq <= 0) | ((q_blk - local_blocks) < 0)
    lo_v = tl.where(any_r, lo, 0)
    hi_v = tl.where(any_r, hi + 1, tl.where(no_remote, 0, w_seq))
    tl.store(lo_ptr + row, lo_v.to(tl.int32))
    tl.store(hi_ptr + row, hi_v.to(tl.int32))
    tl.store(fit_ptr + row, ((hi_v - lo_v) <= tile).to(tl.int8))


@torch.no_grad()
def win_bounds_decode(
    *,
    allow,
    seq_lens,
    lo,
    hi,
    fit,
    max_block: int,
    block_size: int,
    local_blocks: int,
    decode_query_len: int,
    tile: int,
):
    """Fill ``lo``/``hi``/``fit`` (int32/int32/int8 [nd]) from the page mask.

    ``allow`` is the bool/int8 ``[>= nd, WQ]`` per-decode-token page mask; only the
    first ``nd`` rows are read. Writes in place so the buffers stay capture-stable.
    """
    nd = int(lo.numel())
    if nd <= 0:
        return
    a = allow.view(torch.int8) if allow.dtype == torch.bool else allow
    _win_bounds_kernel[(nd,)](
        a,
        seq_lens,
        lo,
        hi,
        fit,
        a.stride(0),
        int(max_block),
        int(a.shape[1]),
        int(block_size),
        int(local_blocks),
        max(1, int(decode_query_len)),
        int(tile),
        CHUNK=1024,
        num_warps=4,
    )


@triton.jit
def _score_prefill_kernel(
    q_ptr,
    kbar_ptr,
    bt_ptr,
    sl_ptr,
    cu_ptr,
    ctx_ptr,
    r2q_ptr,
    out_ptr,
    stride_q_n,
    stride_kbar_p,
    stride_bt_r,
    stride_out_n,
    W,
    inv_sqrt_d,
    block_size,
    local_blocks,
    row_off,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Compute prefill remote scores by query chunk.

    ``row_off`` maps chunk-local rows to absolute queries; candidates must be
    causal-remote and within the live sequence.
    """
    row = tl.program_id(0) + row_off
    c0 = tl.program_id(1) * CHUNK
    r = tl.load(r2q_ptr + row)

    seq = tl.load(sl_ptr + r).to(tl.int32)
    q0 = tl.load(cu_ptr + r).to(tl.int32)
    ctx = tl.load(ctx_ptr + r).to(tl.int32)
    q_pos = ctx + (row - q0)
    q_blk = q_pos // block_size

    off_w = c0 + tl.arange(0, CHUNK)
    in_w = off_w < W
    # A chunk beyond the causal or sequence bound can only contain sentinels.
    if c0 > q_blk - local_blocks or c0 * block_size >= seq:
        tl.store(
            out_ptr + tl.program_id(0) * stride_out_n + off_w,
            tl.full((CHUNK,), _SCORE_NEG, tl.float32),
            mask=in_w,
        )
        return
    keep = in_w & (off_w <= (q_blk - local_blocks)) & (off_w * block_size < seq)

    off_d = tl.arange(0, D)
    pages = tl.load(bt_ptr + r * stride_bt_r + off_w, mask=in_w, other=0).to(tl.int64)
    kb = tl.load(
        kbar_ptr + pages[:, None] * stride_kbar_p + off_d[None, :],
        mask=keep[:, None],
        other=0.0,
    ).to(tl.float32)  # [CHUNK, D]
    q = tl.load(q_ptr + row * stride_q_n + off_d).to(tl.float32)  # [D]
    z = tl.sum(kb * q[None, :], axis=1) * inv_sqrt_d
    z = tl.minimum(tl.maximum(z, -_ACT_CLAMP), _ACT_CLAMP)
    # `out` is the CHUNK-LOCAL score buffer -> row index without ``row_off``.
    tl.store(
        out_ptr + tl.program_id(0) * stride_out_n + off_w,
        tl.where(keep, z, _SCORE_NEG),
        mask=in_w,
    )


@torch.no_grad()
def kbar_all_plan(
    *,
    block_table,
    seq_lens,
    context_lens,
    block_size: int,
    max_block: int,
    max_query_len: int,
    device,
    out=None,
):
    """Plan pages changed by this prefill step: the mutable boundary page plus at
    most one page past the query span. Zero-token entries are ignored; ``out``
    provides fixed graph storage.
    """
    sl = seq_lens.to(device=device, dtype=torch.int64)
    ctx = context_lens.to(device=device, dtype=torch.int64)
    W = min(int(max_block), int(block_table.shape[1]))
    width = min(W, -(-int(max_query_len) // int(block_size)) + 1)
    R = sl.shape[0]
    p0 = ctx // block_size  # [R]
    blk = p0[:, None] + torch.arange(width, device=device)[None, :]
    # blk > W-1 implies blk >= nb (nb <= W), so `tok` is 0 there and the clamped
    # page id is never actually summarized.
    pages = block_table[:, :W].to(torch.int64).gather(1, blk.clamp(max=W - 1))
    tok = (sl[:, None] - blk * block_size).clamp(0, block_size)
    n = R * width
    if out is not None:
        pages_buf, tok_buf = out
        assert pages_buf.shape[0] >= n and tok_buf.shape[0] >= n
        pages_buf[:n].copy_(pages.reshape(-1))
        tok_buf[:n].copy_(tok.reshape(-1).to(torch.int32))
        return (pages_buf[:n], tok_buf[:n])
    return (
        pages.reshape(-1).contiguous(),
        tok.reshape(-1).to(torch.int32).contiguous(),
    )


@torch.no_grad()
def prefill_row2req(*, cu_seqlens_q, device, out=None, total_q=None):
    """Map prefill query rows to requests.

    ``out`` provides fixed graph storage. ``repeat_interleave`` synchronizes for
    sizing; replacing it requires a host-known live row count for padded graphs.
    """
    cu = cu_seqlens_q.to(device=device, dtype=torch.int64)
    q_lens = cu[1:] - cu[:-1]
    r = torch.arange(q_lens.numel(), device=device)
    row2req = torch.repeat_interleave(r, q_lens).to(torch.int32)
    if out is not None:
        n = row2req.shape[0]
        assert out.shape[0] >= n
        out[:n].copy_(row2req)
        return out[:n]
    return row2req.contiguous()


def _launch_entmax(
    *,
    rows: int,
    scores: torch.Tensor,
    img: torch.Tensor,
    has_img: bool,
    base: torch.Tensor,
    fit: torch.Tensor,
    out_idx: torch.Tensor,
    out_bias: torch.Tensor,
    W: int,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    budget: int,
    M: int,
    block_w: int,
    has_win: bool,
    fit_sel: int,
    rel: bool,
) -> None:
    _entmax_kernel[(rows,)](
        scores,
        img,
        base,
        fit,
        out_idx,
        out_bias,
        scores.stride(0),
        img.stride(0) if has_img else 0,
        out_idx.stride(0),
        out_bias.stride(0),
        W,
        idx_scale,
        prior_floor,
        1.0 / sigma,
        budget,
        M=M,
        BLOCK_W=block_w,
        HAS_IMG=has_img,
        HAS_WIN=has_win,
        FIT_SEL=fit_sel,
        REL=rel,
        num_warps=4,
    )


def _launch_prefill_scores(
    *,
    rows: int,
    row_off: int,
    idx_q: torch.Tensor,
    k_bar: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    row2req: torch.Tensor,
    out: torch.Tensor,
    W: int,
    block_size: int,
    local_blocks: int,
) -> None:
    d = idx_q.shape[-1]
    _score_prefill_kernel[(rows, triton.cdiv(W, _SCORE_CHUNK))](
        idx_q,
        k_bar,
        block_table,
        seq_lens,
        cu_seqlens_q,
        context_lens,
        row2req,
        out,
        idx_q.stride(0),
        k_bar.stride(0),
        block_table.stride(0),
        out.stride(0),
        W,
        1.0 / (d**0.5),
        block_size,
        local_blocks,
        row_off,
        D=d,
        CHUNK=_SCORE_CHUNK,
        num_warps=4,
    )


def _check_out_slots(out_slots: int | None, topm: int, out_idx: torch.Tensor) -> int:
    M = min(int(out_slots or topm), int(out_idx.shape[-1]))
    assert M & (M - 1) == 0, f"out_slots must be a power of two, got {M}"
    return M


@torch.no_grad()
def route_prefill_triton(
    *,
    idx_q,
    k_bar,
    block_table,
    seq_lens,
    cu_seqlens_q,
    context_lens,
    row2req,
    image_allow,
    out_idx,
    out_bias,
    max_block: int,
    block_size: int,
    local_blocks: int,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    topm: int,
    budget: int | None = None,
    q_chunk: int = 2048,
    out_slots: int | None = None,
    score_buf: torch.Tensor | None = None,
):
    """Fused chunked prefill routing into caller-owned output buffers.

    ``out_slots`` changes only the packed output width; entmax still evaluates
    ``topm`` candidates. The return contains the final score chunk for diagnostics.
    """
    total_q = int(idx_q.shape[0])
    W = min(int(max_block), int(block_table.shape[1]))
    M = _check_out_slots(out_slots, topm, out_idx)
    QC = max(1, min(int(q_chunk), total_q))

    # Reuse caller-owned scratch during CUDA Graph capture.
    if score_buf is not None:
        assert score_buf.dtype == torch.float32
        sc = score_buf
    else:
        sc = torch.empty(QC, W, dtype=torch.float32, device=idx_q.device)

    has_img = image_allow is not None
    img_all = image_allow.contiguous() if has_img else None
    budget_c = max(1, min(int(budget or M), M))
    off = 0
    for off in range(0, total_q, QC):
        n = min(QC, total_q - off)
        sc_v = sc[:n]
        img = img_all[off : off + n] if has_img else sc_v  # dummy ptr when unused
        _launch_prefill_scores(
            rows=n,
            row_off=off,
            idx_q=idx_q,
            k_bar=k_bar,
            block_table=block_table,
            seq_lens=seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            context_lens=context_lens,
            row2req=row2req,
            out=sc_v,
            W=W,
            block_size=block_size,
            local_blocks=local_blocks,
        )
        _launch_entmax(
            rows=n,
            scores=sc_v,
            img=img,
            has_img=has_img,
            base=sc_v,
            fit=sc_v,
            out_idx=out_idx[off : off + n],
            out_bias=out_bias[off : off + n],
            W=W,
            idx_scale=idx_scale,
            prior_floor=prior_floor,
            sigma=sigma,
            budget=budget_c,
            M=M,
            block_w=triton.next_power_of_2(W),
            has_win=False,
            fit_sel=0,
            rel=False,
        )
    return M, sc[: min(QC, total_q - off)], off


@torch.no_grad()
def kbar_tail_plan(
    *, block_table, seq_lens, block_size: int, tail_pages: int, device, out=None
):
    """Plan mutable tail pages and live token counts.

    All DASH layers share the plan; ``out`` provides fixed graph storage.
    """
    sl = seq_lens.to(device=device, dtype=torch.int64)
    nb = (sl + block_size - 1) // block_size
    last = (nb - 1).clamp(min=0)
    offs = torch.arange(tail_pages, device=device)
    blk = (last[:, None] - offs[None, :]).clamp(min=0)
    w = int(block_table.shape[1])
    pages = block_table[:, :w].to(torch.int64).gather(1, blk.clamp(max=w - 1))
    tok = (sl[:, None] - blk * block_size).clamp(0, block_size)
    n = pages.numel()
    if out is None or out[0].shape[0] < n:
        return (
            pages.reshape(-1).contiguous(),
            tok.reshape(-1).to(torch.int32).contiguous(),
        )
    p_buf, t_buf = out[0][:n], out[1][:n]
    p_buf.copy_(pages.reshape(-1))
    t_buf.copy_(tok.reshape(-1))
    return p_buf, t_buf


@torch.no_grad()
def kbar_refresh_tail_triton(
    *,
    kv_cache,
    k_bar,
    block_table,
    seq_lens,
    head_cls,
    block_size: int,
    tail_pages: int = 1,
    plan=None,
):
    """Refresh the last ``tail_pages`` page summaries of every request."""
    device = k_bar.device
    d = k_bar.shape[-1]
    if plan is None:
        plan = kbar_tail_plan(
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=block_size,
            tail_pages=tail_pages,
            device=device,
        )
    pages, tok = plan

    _kbar_page_kernel[(pages.numel(),)](
        kv_cache,
        head_cls,
        tok,
        pages,
        k_bar,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        k_bar.stride(0),
        1.0 / (d**0.5),
        D=d,
        BS=block_size,
        num_warps=4,
    )


class _DecodeRoutePlan(NamedTuple):
    """Launch shape of one decode routing call, fixed by buffer geometry."""

    M: int
    bw_full: int
    ent_win: bool
    tile: int
    rel: bool
    tiled: bool


def _plan_decode_route(
    *,
    nd: int,
    W: int,
    out_idx: torch.Tensor,
    topm: int,
    out_slots: int | None,
    entmax_base,
    entmax_fit,
    entmax_width: int,
    score_buf,
    score_buf_win,
    bounds,
) -> _DecodeRoutePlan:
    M = _check_out_slots(out_slots, topm, out_idx)
    bw_full = triton.next_power_of_2(W)
    ent_win = (
        entmax_base is not None
        and entmax_fit is not None
        and 0 < int(entmax_width) < bw_full
    )
    if ent_win:
        assert int(entmax_width) & (int(entmax_width) - 1) == 0, (
            f"entmax_width must be a power of two, got {entmax_width}"
        )
    tile = int(entmax_width) if ent_win else 0
    # TILE must be a whole number of CHUNKs: the narrow grid is cdiv(TILE, CHUNK)
    # and each program stores CHUNK lanes, so a partial chunk would overrun the row.
    rel = (
        ent_win
        and score_buf_win is not None
        and tile % _SCORE_CHUNK == 0
        and score_buf_win.shape[0] >= nd
        and score_buf_win.shape[1] >= tile
    )
    tiled = (
        _want_tiled(bw_full)
        and ent_win
        and bounds is not None
        and tile % _SCORE_CHUNK == 0
        and score_buf is not None
        and score_buf.shape[0] >= nd
        and score_buf.shape[1] >= W
    )
    return _DecodeRoutePlan(
        M=M, bw_full=bw_full, ent_win=ent_win, tile=tile, rel=rel, tiled=tiled
    )


def _launch_decode_scores(
    *,
    idx_q: torch.Tensor,
    k_bar: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    lo: torch.Tensor,
    fit: torch.Tensor,
    W: int,
    width: int,
    block_size: int,
    local_blocks: int,
    dql: int,
    rel: bool,
    fit_sel: int,
) -> None:
    d = idx_q.shape[-1]
    _score_kernel[(int(idx_q.shape[0]), triton.cdiv(width, _SCORE_CHUNK))](
        idx_q,
        k_bar,
        block_table,
        seq_lens,
        out,
        lo,
        fit,
        idx_q.stride(0),
        k_bar.stride(0),
        block_table.stride(0),
        out.stride(0),
        W,
        1.0 / (d**0.5),
        block_size,
        local_blocks,
        dql,
        D=d,
        CHUNK=_SCORE_CHUNK,
        REL=rel,
        FIT_SEL=fit_sel,
        num_warps=4,
    )


def _launch_scan_decode_route(
    *,
    plan: _DecodeRoutePlan,
    idx_q,
    k_bar,
    block_table,
    seq_lens,
    sc,
    img,
    has_img: bool,
    entmax_base,
    entmax_fit,
    score_buf_win,
    out_idx,
    out_bias,
    W: int,
    block_size: int,
    local_blocks: int,
    dql: int,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    budget: int | None,
) -> None:
    """Route decode rows by a full-width scan after an optional narrow pass."""
    nd = int(idx_q.shape[0])
    ent_base = entmax_base if plan.ent_win else sc
    ent_fit = entmax_fit if plan.ent_win else sc
    scores = partial(
        _launch_decode_scores,
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=block_table,
        seq_lens=seq_lens,
        W=W,
        block_size=block_size,
        local_blocks=local_blocks,
        dql=dql,
    )
    entmax = partial(
        _launch_entmax,
        rows=nd,
        img=img,
        has_img=has_img,
        base=ent_base,
        fit=ent_fit,
        out_idx=out_idx,
        out_bias=out_bias,
        W=W,
        idx_scale=idx_scale,
        prior_floor=prior_floor,
        sigma=sigma,
        budget=max(1, min(int(budget or plan.M), plan.M)),
        M=plan.M,
    )
    if plan.rel:
        # Narrow, window-relative: the only launch that scales with the page
        # span instead of the block-table width.
        sc_win = score_buf_win[:nd, : plan.tile]
        scores(
            out=sc_win,
            lo=entmax_base,
            fit=entmax_fit,
            width=plan.tile,
            rel=True,
            fit_sel=1,
        )
        entmax(scores=sc_win, block_w=plan.tile, has_win=True, fit_sel=1, rel=True)

    # The full-width companion is required for rows outside the 128-column window.
    scores(
        out=sc,
        lo=ent_base,
        fit=entmax_fit if plan.rel else sc,
        width=W,
        rel=False,
        fit_sel=2 if plan.rel else 0,
    )
    if not plan.rel:
        entmax(
            scores=sc,
            block_w=plan.tile if plan.ent_win else plan.bw_full,
            has_win=plan.ent_win,
            fit_sel=1 if plan.ent_win else 0,
            rel=False,
        )
    if plan.ent_win:
        # Always launched so graph capture is input-independent; programs return
        # immediately for rows the narrow tile already wrote.
        entmax(scores=sc, block_w=plan.bw_full, has_win=False, fit_sel=2, rel=False)


@torch.no_grad()
def route_decode_triton(
    *,
    idx_q,
    k_bar,
    block_table,
    seq_lens,
    image_allow,
    out_idx,
    out_bias,
    max_block: int,
    block_size: int,
    local_blocks: int,
    decode_query_len: int,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    topm: int,
    score_buf=None,
    budget: int | None = None,
    out_slots: int | None = None,
    entmax_base=None,
    entmax_fit=None,
    entmax_width: int = 0,
    score_buf_win=None,
    bounds=None,
):
    """Fused decode routing into caller-owned indices and biases.

    ``out_slots`` controls only packed output width and ``budget`` the hard top-k
    selection; optional bounds use narrow or tiled scans without changing entmax.
    """
    dql = max(1, int(decode_query_len))
    nd = int(idx_q.shape[0])
    W = min(int(max_block), int(block_table.shape[1]))
    plan = _plan_decode_route(
        nd=nd,
        W=W,
        out_idx=out_idx,
        topm=topm,
        out_slots=out_slots,
        entmax_base=entmax_base,
        entmax_fit=entmax_fit,
        entmax_width=entmax_width,
        score_buf=score_buf,
        score_buf_win=score_buf_win,
        bounds=bounds,
    )
    if score_buf is not None and score_buf.shape[0] >= nd and score_buf.shape[1] >= W:
        sc = score_buf[:nd, :W]
    else:
        sc = torch.empty(nd, W, dtype=torch.float32, device=idx_q.device)
    has_img = image_allow is not None
    img = image_allow.contiguous() if has_img else sc  # dummy ptr when unused

    if plan.tiled:
        _launch_tiled_decode_route(
            idx_q=idx_q,
            k_bar=k_bar,
            block_table=block_table,
            seq_lens=seq_lens,
            sc_rel=sc,
            img=img,
            has_img=has_img,
            bounds=bounds,
            out_idx=out_idx,
            out_bias=out_bias,
            W=W,
            M=plan.M,
            TILE=plan.tile,
            CHUNK=_SCORE_CHUNK,
            block_size=block_size,
            local_blocks=local_blocks,
            dql=dql,
            idx_scale=idx_scale,
            prior_floor=prior_floor,
            sigma=sigma,
            budget=budget,
        )
        return plan.M

    _launch_scan_decode_route(
        plan=plan,
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=block_table,
        seq_lens=seq_lens,
        sc=sc,
        img=img,
        has_img=has_img,
        entmax_base=entmax_base,
        entmax_fit=entmax_fit,
        score_buf_win=score_buf_win,
        out_idx=out_idx,
        out_bias=out_bias,
        W=W,
        block_size=block_size,
        local_blocks=local_blocks,
        dql=dql,
        idx_scale=idx_scale,
        prior_floor=prior_floor,
        sigma=sigma,
        budget=budget,
    )
    return plan.M


def _launch_tiled_decode_route(
    *,
    idx_q,
    k_bar,
    block_table,
    seq_lens,
    sc_rel,
    img,
    has_img: bool,
    bounds,
    out_idx,
    out_bias,
    W: int,
    M: int,
    TILE: int,
    CHUNK: int,
    block_size: int,
    local_blocks: int,
    dql: int,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    budget: int | None,
) -> None:
    """Score and entmax-route decode queries over bounded tiles of the page scan."""
    d = idx_q.shape[-1]
    nd = int(idx_q.shape[0])
    b_lo, b_hi = bounds
    _score_tiled_kernel[(nd, triton.cdiv(TILE, CHUNK))](
        idx_q,
        k_bar,
        block_table,
        seq_lens,
        sc_rel,
        b_lo,
        b_hi,
        idx_q.stride(0),
        k_bar.stride(0),
        block_table.stride(0),
        sc_rel.stride(0),
        W,
        1.0 / (d**0.5),
        block_size,
        local_blocks,
        dql,
        D=d,
        CHUNK=CHUNK,
        TILE=TILE,
        num_warps=4,
    )
    _entmax_tiled_kernel[(nd,)](
        sc_rel,
        img,
        b_lo,
        b_hi,
        out_idx,
        out_bias,
        sc_rel.stride(0),
        img.stride(0) if has_img else 0,
        out_idx.stride(0),
        out_bias.stride(0),
        W,
        idx_scale,
        prior_floor,
        1.0 / sigma,
        max(1, min(int(budget or M), M)),
        M=M,
        TILE=TILE,
        HAS_IMG=has_img,
        num_warps=4,
    )
