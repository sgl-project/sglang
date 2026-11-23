# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Stands in for the stock recurrent GDN decode and verify kernels (fused_recurrent,
fused_sigmoid_gating_delta_rule_update), which advance the FP32 state one token at a time; that associates differently
from the chunk-64 prefill, and no kernel configuration aligns the two.

TOP chunked-GDN row engine: decode and chain speculative verify on FLA's chunk-64 program.

Same contract as chunk_rows.py (each token's output and the FP32 boundary states are bitwise those of the fixed-tile
FLA chunk prefill), generalized to up to NMAX consecutive rows per request per step and a deferred chunk close:

- a step runs rows p .. p + n - 1 (n <= NMAX) of one 64-token chunk for each request (decode: n = 1; chain verify:
  the root token plus the drafts, the window cut at row 63);
- the slot's boundary state E advances over a chunk only at the first step of the next chunk (`e_chunk[slot, head]`
  is the chunk E stands at), so a verify never writes E: rows of rejected drafts are overwritten by later steps and
  nothing has to be rolled back. A committed length is just the position the next step starts at.

Per layer per step (fixed grids, CUDA-graph capturable): `_close_pending_kernel` (B, H) closes a finished chunk;
`_prep_kernel` (B, H) l2-normalizes q and k, gates, writes the rows and runs the chunk cumsum; `_solve_kernel` (B * H) the rows of the (at most two)
16-row blocks of (I + A)^{-1} the window touches; `_wu_kernel` (B * H) rows of w and u; `_out_kernel` (V / BV, B * H)
rows of v_new and of the output, on a 32-row tile from the window's first 16-row block.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.chunk_rows import (
    _blk,
    _kk,
    _mm,
    _put,
)
from sglang.kernels.ops.attention.fla.op import exp, safe_exp

CHUNK = tl.constexpr(64)
TILE = tl.constexpr(
    32
)  # rows p .. p + NMAX - 1 always lie in rows 16 * (p // 16) .. + 31 for NMAX <= 16


@triton.jit
def _close_head(
    state,
    ss,
    dst,
    ds,
    k_rows,
    v_rows,
    beta_rows,
    g_rows,
    sv,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """The state kernel's step over a complete chunk for one head, from the chunk's cached rows: E at
    state[ss] (head base) advanced over the chunk into dst[ds] (in place when they are the same).

    Mirrors chunk_gated_delta_rule_fwd_kernel_h_blockdim64 op for op (K = 128 as two 64-wide halves of the
    [BV, 64] state: v_new = u - (w_1 bf16(h_1)^T + w_2 bf16(h_2)^T), then h_i = fma(h_i, exp(g_last),
    (k_i^T v_new)^T)): with a decay near 1 the decayed state is not small, and rounding the product before the add
    changes its last bit."""
    tl.static_assert(K == 128)
    r64 = tl.arange(0, CHUNK)
    c64 = tl.arange(0, CHUNK)
    b_s = tl.load(g_rows + r64 * H).to(tl.float32)
    b_G = tl.cumsum(b_s, axis=0)
    b_beta = tl.load(beta_rows + r64 * H)
    b_A = tl.load(sv + r64[:, None] * CHUNK + c64[None, :]).to(tl.bfloat16)
    b_eg = tl.exp(b_G)
    # recompute_w_u: w = A @ bf16(k * beta * exp(g)) per 64 columns of K.
    b_k1 = tl.load(k_rows + r64[:, None] * (Hg * K) + c64[None, :])
    b_k2 = tl.load(k_rows + r64[:, None] * (Hg * K) + 64 + c64[None, :])
    b_w1 = tl.dot(b_A, (b_k1 * b_beta[:, None] * b_eg[:, None]).to(b_k1.dtype)).to(
        tl.bfloat16
    )
    b_w2 = tl.dot(b_A, (b_k2 * b_beta[:, None] * b_eg[:, None]).to(b_k2.dtype)).to(
        tl.bfloat16
    )
    # k^T halves, loaded transposed as the state kernel does ([64 of K, BT]).
    b_kT1 = tl.load(k_rows + r64[None, :] * (Hg * K) + c64[:, None])
    b_kT2 = tl.load(k_rows + r64[None, :] * (Hg * K) + 64 + c64[:, None])
    g_last = tl.sum(tl.where(r64 == CHUNK - 1, b_G, 0.0), 0)
    vdec = safe_exp(g_last - b_G)
    decay = exp(g_last)
    h0 = state + (ss * H) * V * K
    h1 = dst + (ds * H) * V * K
    for i_v in tl.static_range(V // BV):
        cv = i_v * BV + tl.arange(0, BV)
        b_v = tl.load(v_rows + r64[:, None] * (H * V) + cv[None, :])
        b_u = tl.dot(b_A, (b_v * b_beta[:, None]).to(b_v.dtype), allow_tf32=False).to(
            tl.bfloat16
        )
        b_h1 = tl.load(h0 + cv[:, None] * K + c64[None, :])
        b_h2 = tl.load(h0 + cv[:, None] * K + 64 + c64[None, :])
        c_v = tl.dot(b_w1, tl.trans(b_h1).to(tl.bfloat16))
        c_v += tl.dot(b_w2, tl.trans(b_h2).to(tl.bfloat16))
        c_v = b_u - c_v
        c_v = c_v * vdec[:, None]
        c_vb = c_v.to(tl.bfloat16)
        # The state kernel's `h *= exp(g_last); h += (k^T v)^T` compiles to one FMA per element (a single
        # rounding); spelled out here so the close does not depend on the compiler contracting it the same way.
        b_h1 = tl.fma(b_h1, decay, tl.trans(tl.dot(b_kT1, c_vb)))
        b_h2 = tl.fma(b_h2, decay, tl.trans(tl.dot(b_kT2, c_vb)))
        tl.store(h1 + cv[:, None] * K + c64[None, :], b_h1)
        tl.store(h1 + cv[:, None] * K + 64 + c64[None, :], b_h2)


@triton.jit
def _close_pending_kernel(
    state,
    k_ring,
    v_ring,
    g_ring,
    beta_ring,
    solved,
    e_chunk,
    slots,
    state_slots,
    rows,
    counts,
    chunks,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    """The deferred close: a step starting a new chunk (row 0) first advances E over the previous chunk when
    e_chunk says E still stands at an earlier one; grid (B, H). Its own launch (compiled as close_to), not part of
    the prep kernel, so its arithmetic does not depend on the step's shape."""
    i_n, i_h = tl.program_id(0), tl.program_id(1)
    s = tl.load(slots + i_n).to(tl.int64)
    p = tl.load(rows + i_n)
    n = tl.load(counts + i_n)
    c = tl.load(chunks + i_n)
    ss = tl.load(state_slots + i_n)
    e_ptr = e_chunk + s * H + i_h
    if (n > 0) & (p == 0) & (ss >= 0):
        if tl.load(e_ptr) < c:
            ss64 = ss.to(tl.int64)
            _close_head(
                state + i_h * V * K,
                ss64,
                state + i_h * V * K,
                ss64,
                k_ring + (s * CHUNK * Hg + i_h // (H // Hg)) * K,
                v_ring + (s * CHUNK * H + i_h) * V,
                beta_ring + s * CHUNK * H + i_h,
                g_ring + s * CHUNK * H + i_h,
                solved + (s * H + i_h) * CHUNK * CHUNK,
                H,
                Hg,
                K,
                V,
                32,
            )
            tl.store(e_ptr, c)


@triton.jit
def _prep_kernel(
    qkv,
    stride_qkv,
    a,
    b,
    A_log,
    dt_bias,
    k_ring,
    v_ring,
    g_ring,
    beta_ring,
    q_out,
    g_cum,
    slots,
    rows,
    counts,
    eps,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NMAX: tl.constexpr,
):
    i_n, i_h = tl.program_id(0), tl.program_id(1)
    s = tl.load(slots + i_n).to(tl.int64)
    p = tl.load(rows + i_n)
    n = tl.load(counts + i_n)
    r16 = tl.arange(0, 16)
    ck = tl.arange(0, K)
    cv = tl.arange(0, V)
    for j in tl.static_range(NMAX):
        if j < n:
            x = qkv + (i_n * NMAX + j).to(tl.int64) * stride_qkv
            r = p + j
            if i_h == 0:
                # l2norm_fwd_kernel's tile: 16 rows (one token's Hg = 16 key heads) x K, 8 warps.
                for which in tl.static_range(2):
                    b_x = tl.load(
                        x + which * Hg * K + r16[:, None] * K + ck[None, :]
                    ).to(tl.float32)
                    b_var = tl.sum(b_x * b_x, axis=1)
                    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
                    if which == 0:
                        dst = (
                            q_out
                            + ((i_n * NMAX + j) * Hg + r16[:, None]) * K
                            + ck[None, :]
                        )
                    else:
                        dst = (
                            k_ring
                            + ((s * CHUNK + r) * Hg + r16[:, None]) * K
                            + ck[None, :]
                        )
                    tl.store(dst, b_y.to(tl.bfloat16))
            tl.store(
                v_ring + ((s * CHUNK + r) * H + i_h) * V + cv,
                tl.load(x + 2 * Hg * K + i_h * V + cv),
            )
            # fused_gdn_gating_kernel (beta = 1, threshold = 20); beta rounded to b's dtype as there.
            t = (i_n * NMAX + j) * H + i_h
            blk_x = tl.load(a + t).to(tl.float32) + tl.load(dt_bias + i_h).to(
                tl.float32
            )
            softplus_x = tl.where(
                1.0 * blk_x <= 20.0, (1 / 1.0) * tl.log(1 + tl.exp(1.0 * blk_x)), blk_x
            )
            blk_g = -tl.exp(tl.load(A_log + i_h).to(tl.float32)) * softplus_x
            blk_b = tl.load(b + t)
            blk_beta = tl.sigmoid(blk_b.to(tl.float32)).to(blk_b.dtype).to(tl.float32)
            tl.store(g_ring + (s * CHUNK + r) * H + i_h, blk_g)
            tl.store(beta_ring + (s * CHUNK + r) * H + i_h, blk_beta)
    tl.debug_barrier()
    # chunk_local_cumsum_scalar_kernel over the partial chunk ending at the window's last row.
    r64 = tl.arange(0, CHUNK)
    b_s = tl.load(g_ring + (s * CHUNK + r64) * H + i_h, mask=r64 < p + n, other=0.0).to(
        tl.float32
    )
    tl.store(g_cum + (i_n * CHUNK + r64) * H + i_h, tl.cumsum(b_s, axis=0))


@triton.jit
def _solve_block(
    k_rows,
    beta_ring,
    s,
    g_row,
    sv,
    cu,
    J,
    last,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    cu_off,
):
    """Rows <= last of 16-row block J of (I + A)^{-1} into cur rows cu_off .. + 15; the block into the solved cache
    when last completes it."""
    o_i = tl.arange(0, 16)
    rJ = J * 16 + o_i
    b_gJ = tl.load(g_row + rJ * H)
    b_bJ = tl.load(beta_ring + (s * CHUNK + rJ) * H, mask=rJ <= last, other=0.0)
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    m_J = rJ <= last
    b_A = _kk(k_rows, J, J, last, Hg, K, BK)
    b_A *= safe_exp(b_gJ[:, None] - b_gJ[None, :])
    b_A = tl.where(m_d & (m_J[:, None] & m_J[None, :]), b_A, 0.0) * b_bJ[:, None]
    b_Ai = -b_A
    for i in range(2, tl.minimum(16, last % 16 + 1)):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A, 0.0), 0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    cr = cu + cu_off * CHUNK
    _put(cr, 0, J, b_Ai, CHUNK)
    done = last % 16 == 15
    if done:
        _put(sv, J, J, b_Ai, CHUNK)
    gate = b_bJ[:, None]
    if J >= 1:
        b_g0 = tl.load(g_row + o_i * H)
        b_A0 = _kk(k_rows, J, 0, last, Hg, K, BK) * safe_exp(
            b_gJ[:, None] - b_g0[None, :]
        )
        b_A0 = b_A0 * gate
        b_Ai00 = _blk(sv, 0, 0, CHUNK)
        if J == 1:
            b_Ai10 = -_mm(_mm(b_Ai, b_A0), b_Ai00)
            _put(cr, 0, 0, b_Ai10, CHUNK)
            if done:
                _put(sv, 1, 0, b_Ai10, CHUNK)
        else:
            b_g1 = tl.load(g_row + (16 + o_i) * H)
            b_A1 = _kk(k_rows, J, 1, last, Hg, K, BK) * safe_exp(
                b_gJ[:, None] - b_g1[None, :]
            )
            b_A1 = b_A1 * gate
            b_Ai11 = _blk(sv, 1, 1, CHUNK)
            b_Ai10 = _blk(sv, 1, 0, CHUNK)
            if J == 2:
                b_Ai21 = -_mm(_mm(b_Ai, b_A1), b_Ai11)
                b_Ai20 = -_mm(b_Ai, _mm(b_A0, b_Ai00) + _mm(b_A1, b_Ai10))
                _put(cr, 0, 1, b_Ai21, CHUNK)
                _put(cr, 0, 0, b_Ai20, CHUNK)
                if done:
                    _put(sv, 2, 1, b_Ai21, CHUNK)
                    _put(sv, 2, 0, b_Ai20, CHUNK)
            else:
                b_g2 = tl.load(g_row + (32 + o_i) * H)
                b_A2 = _kk(k_rows, J, 2, last, Hg, K, BK) * safe_exp(
                    b_gJ[:, None] - b_g2[None, :]
                )
                b_A2 = b_A2 * gate
                b_Ai22 = _blk(sv, 2, 2, CHUNK)
                b_Ai21 = _blk(sv, 2, 1, CHUNK)
                b_Ai20 = _blk(sv, 2, 0, CHUNK)
                b_Ai32 = -_mm(_mm(b_Ai, b_A2), b_Ai22)
                b_Ai31 = -_mm(b_Ai, _mm(b_A1, b_Ai11) + _mm(b_A2, b_Ai21))
                b_Ai30 = -_mm(
                    b_Ai, _mm(b_A0, b_Ai00) + _mm(b_A1, b_Ai10) + _mm(b_A2, b_Ai20)
                )
                _put(cr, 0, 2, b_Ai32, CHUNK)
                _put(cr, 0, 1, b_Ai31, CHUNK)
                _put(cr, 0, 0, b_Ai30, CHUNK)
                if done:
                    _put(sv, 3, 2, b_Ai32, CHUNK)
                    _put(sv, 3, 1, b_Ai31, CHUNK)
                    _put(sv, 3, 0, b_Ai30, CHUNK)


@triton.jit
def _solve_kernel(
    k_ring,
    beta_ring,
    g_cum,
    solved,
    cur,
    slots,
    rows,
    counts,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    i_n, i_h = tl.program_id(0) // H, tl.program_id(0) % H
    n = tl.load(counts + i_n)
    if n > 0:
        s = tl.load(slots + i_n).to(tl.int64)
        p = tl.load(rows + i_n)
        last = p + n - 1
        J0 = p // 16
        k_rows = k_ring + (s * CHUNK * Hg + i_h // (H // Hg)) * K
        g_row = g_cum + i_n * CHUNK * H + i_h
        sv = solved + (s * H + i_h) * CHUNK * CHUNK
        cu = cur + (i_n * H + i_h) * TILE * CHUNK
        r32 = tl.arange(0, TILE)
        c64 = tl.arange(0, CHUNK)
        tl.store(
            cu + r32[:, None] * CHUNK + c64[None, :],
            tl.zeros([TILE, CHUNK], tl.float32),
        )
        tl.debug_barrier()
        _solve_block(
            k_rows,
            beta_ring + i_h,
            s,
            g_row,
            sv,
            cu,
            J0,
            tl.minimum(last, J0 * 16 + 15),
            H,
            Hg,
            K,
            BK,
            0,
        )
        if last // 16 > J0:
            tl.debug_barrier()
            _solve_block(
                k_rows,
                beta_ring + i_h,
                s,
                g_row,
                sv,
                cu,
                J0 + 1,
                last,
                H,
                Hg,
                K,
                BK,
                16,
            )


@triton.jit
def _wu_kernel(
    k_ring,
    v_ring,
    beta_ring,
    g_cum,
    cur,
    w_row,
    u_row,
    slots,
    rows,
    counts,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NMAX: tl.constexpr,
):
    """Rows p .. p + n - 1 of w and u (recompute_w_u arithmetic) on the 32-row tile."""
    i_n, i_h = tl.program_id(0) // H, tl.program_id(0) % H
    n = tl.load(counts + i_n)
    if n > 0:
        s = tl.load(slots + i_n).to(tl.int64)
        p = tl.load(rows + i_n)
        last = p + n - 1
        k_rows = k_ring + (s * CHUNK * Hg + i_h // (H // Hg)) * K
        cu = cur + (i_n * H + i_h) * TILE * CHUNK
        base = p // 16 * 16
        r_t = base + tl.arange(0, TILE)
        is_r = (r_t >= p) & (r_t <= last)
        j_t = tl.where(is_r, r_t - p, 0)
        c64 = tl.arange(0, CHUNK)
        b_Ar = tl.load(
            cu + tl.arange(0, TILE)[:, None] * CHUNK + c64[None, :],
            mask=is_r[:, None],
            other=0.0,
        )
        b_Ar = b_Ar.to(tl.bfloat16)
        r64 = tl.arange(0, CHUNK)
        m_r = (r64 <= last)[:, None]
        b_beta = tl.load(
            beta_ring + (s * CHUNK + r64) * H + i_h, mask=r64 <= last, other=0.0
        )
        cv = tl.arange(0, V)
        b_v = tl.load(
            v_ring + (s * CHUNK * H + i_h) * V + r64[:, None] * (H * V) + cv[None, :],
            mask=m_r,
            other=0.0,
        )
        b_u = tl.dot(b_Ar, (b_v * b_beta[:, None]).to(b_v.dtype), allow_tf32=False)
        tl.store(
            u_row + ((i_n * NMAX + j_t[:, None]) * H + i_h) * V + cv[None, :],
            b_u.to(tl.bfloat16),
            mask=is_r[:, None],
        )
        b_eg = tl.exp(tl.load(g_cum + (i_n * CHUNK + r64) * H + i_h))
        for i_k in tl.static_range(K // 64):
            ck = i_k * 64 + c64
            b_k = tl.load(
                k_rows + r64[:, None] * (Hg * K) + ck[None, :], mask=m_r, other=0.0
            )
            b_w = tl.dot(b_Ar, (b_k * b_beta[:, None] * b_eg[:, None]).to(b_k.dtype))
            tl.store(
                w_row + ((i_n * NMAX + j_t[:, None]) * H + i_h) * K + ck[None, :],
                b_w.to(tl.bfloat16),
                mask=is_r[:, None],
            )


@triton.jit
def _out_kernel(
    state,
    k_ring,
    vnew_ring,
    g_cum,
    w_row,
    u_row,
    q_in,
    out,
    slots,
    state_slots,
    rows,
    counts,
    scale,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    NMAX: tl.constexpr,
):
    """Rows p .. p + n - 1 of v_new (state kernel) and of the output (output kernel) on the 32-row tile."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    n = tl.load(counts + i_n)
    if n > 0:
        i_hg = i_h // (H // Hg)
        s = tl.load(slots + i_n).to(tl.int64)
        ss = tl.maximum(tl.load(state_slots + i_n), 0).to(tl.int64)
        p = tl.load(rows + i_n)
        last = p + n - 1
        base = p // 16 * 16
        r_t = base + tl.arange(0, TILE)
        is_r = (r_t >= p) & (r_t <= last)
        j_t = tl.where(is_r, r_t - p, 0)
        r64 = tl.arange(0, CHUNK)
        cv = i_v * BV + tl.arange(0, BV)
        ck = tl.arange(0, K)
        k_rows = k_ring + (s * CHUNK * Hg + i_hg) * K
        g_rows = g_cum + i_n * CHUNK * H + i_h
        b_G = tl.load(g_rows + r64 * H)
        h0 = state + (ss * H + i_h) * V * K
        b_E = tl.load(h0 + cv[:, None] * K + ck[None, :])
        b_Eb = b_E.to(tl.bfloat16)
        tok = i_n * NMAX + j_t
        b_u = tl.load(
            u_row + (tok[:, None] * H + i_h) * V + cv[None, :],
            mask=is_r[:, None],
            other=0.0,
        )
        b_w = tl.load(
            w_row + (tok[:, None] * H + i_h) * K + ck[None, :],
            mask=is_r[:, None],
            other=0.0,
        )
        b_vn = b_u - tl.dot(b_w, tl.trans(b_Eb))
        vn_rows = vnew_ring + (s * CHUNK * H + i_h) * V
        tl.store(
            vn_rows + r_t[:, None] * (H * V) + cv[None, :],
            b_vn.to(tl.bfloat16),
            mask=is_r[:, None],
        )
        tl.debug_barrier()
        b_q = tl.load(
            q_in + (tok[:, None] * Hg + i_hg) * K + ck[None, :],
            mask=is_r[:, None],
            other=0.0,
        )
        b_kT = tl.load(
            k_rows + r64[None, :] * (Hg * K) + ck[:, None],
            mask=(r64 <= last)[None, :],
            other=0.0,
        )
        b_o = tl.dot(b_q, tl.trans(b_Eb))
        b_A = tl.dot(b_q, b_kT)
        b_gr = tl.load(g_rows + r_t * H, mask=r_t < CHUNK, other=0.0)
        b_o = b_o * exp(b_gr)[:, None]
        b_A = b_A * safe_exp(b_gr[:, None] - b_G[None, :])
        b_A = tl.where(r_t[:, None] >= r64[None, :], b_A, 0)
        b_vt = tl.load(
            vn_rows + r64[:, None] * (H * V) + cv[None, :],
            mask=(r64 <= last)[:, None],
            other=0.0,
        )
        b_o = b_o * scale + tl.dot(b_A.to(b_vt.dtype), b_vt) * scale
        tl.store(
            out + (tok[:, None] * H + i_h) * V + cv[None, :],
            b_o.to(tl.bfloat16),
            mask=is_r[:, None],
        )


@triton.jit
def _close_to_kernel(
    state,
    k_ring,
    v_ring,
    g_ring,
    beta_ring,
    solved,
    slots,
    state_slots,
    dst_slots,
    stride_state_l,
    stride_k_l,
    stride_v_l,
    stride_gb_l,
    stride_solved_l,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    """For each row with dst_slots[n] >= 0: the state at the end of its open chunk (all 64 rows cached) into
    state[dst], E unchanged; grid (B, H, layers)."""
    i_n, i_h, i_l = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    ds = tl.load(dst_slots + i_n).to(tl.int64)
    ss = tl.load(state_slots + i_n).to(tl.int64)
    if (ds >= 0) & (ss >= 0):
        s = tl.load(slots + i_n).to(tl.int64)
        st = state + i_l * stride_state_l + i_h * V * K
        _close_head(
            st,
            ss,
            st,
            ds,
            k_ring + i_l * stride_k_l + (s * CHUNK * Hg + i_h // (H // Hg)) * K,
            v_ring + i_l * stride_v_l + (s * CHUNK * H + i_h) * V,
            beta_ring + i_l * stride_gb_l + s * CHUNK * H + i_h,
            g_ring + i_l * stride_gb_l + s * CHUNK * H + i_h,
            solved + i_l * stride_solved_l + (s * H + i_h) * CHUNK * CHUNK,
            H,
            Hg,
            K,
            V,
            32,
        )


def close_to(*, cache, state, slots, dst_slots, num_layers=None):
    """The state after each row's complete open chunk (rows 0..63 cached, close still deferred) into
    state[..., dst_slots[i]], for rows with dst_slots[i] >= 0, leaving E in place.

    ``cache`` and ``state`` either hold one layer (state [slots, H, V, K]) or stack ``num_layers`` layers on a
    leading dim (one launch for all of them)."""
    ring_slots, state_slots = slots
    B = ring_slots.shape[0]
    if B == 0:
        return
    stacked = num_layers is not None
    L = num_layers if stacked else 1
    H, V, K = state.shape[-3], state.shape[-2], state.shape[-1]
    Hg = cache.k.shape[-2]

    def stride(t):
        return t.stride(0) if stacked else 0

    _close_to_kernel[(B, H, L)](
        state,
        cache.k,
        cache.v,
        cache.g,
        cache.beta,
        cache.solved,
        ring_slots,
        state_slots,
        dst_slots.to(torch.int32),
        stride(state),
        stride(cache.k),
        stride(cache.v),
        stride(cache.g),
        stride(cache.solved),
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        num_warps=4,
    )


class RowCache:
    """Per-layer rows of each slot's open chunk; the last slot absorbs padded rows (slot -1)."""

    def __init__(
        self, *, num_slots, num_k_heads, num_v_heads, head_k_dim, head_v_dim, device
    ):
        assert head_k_dim == 128 and num_k_heads == 16 and head_v_dim % 64 == 0
        n, Hg, H, K, V = num_slots + 1, num_k_heads, num_v_heads, head_k_dim, head_v_dim
        self.pad_slot = num_slots
        self.k = torch.zeros(n, 64, Hg, K, dtype=torch.bfloat16, device=device)
        self.v = torch.zeros(n, 64, H, V, dtype=torch.bfloat16, device=device)
        self.v_new = torch.zeros(n, 64, H, V, dtype=torch.bfloat16, device=device)
        self.g = torch.zeros(n, 64, H, dtype=torch.float32, device=device)
        self.beta = torch.zeros(n, 64, H, dtype=torch.float32, device=device)
        self.solved = torch.zeros(n, H, 64, 64, dtype=torch.float32, device=device)
        self.e_chunk = torch.zeros(n, H, dtype=torch.int32, device=device)

    def bytes_per_slot(self):
        return sum(
            t[0].numel() * t.element_size()
            for t in (
                self.k,
                self.v,
                self.v_new,
                self.g,
                self.beta,
                self.solved,
                self.e_chunk,
            )
        )


class Scratch:
    def __init__(self, max_bs, nmax, Hg, H, K, V, device):
        self.q = torch.empty(max_bs * nmax, Hg, K, dtype=torch.bfloat16, device=device)
        self.g_cum = torch.empty(max_bs, 64, H, dtype=torch.float32, device=device)
        self.cur = torch.empty(max_bs, H, 32, 64, dtype=torch.float32, device=device)
        self.w = torch.empty(max_bs * nmax, H, K, dtype=torch.bfloat16, device=device)
        self.u = torch.empty(max_bs * nmax, H, V, dtype=torch.bfloat16, device=device)


def step_slots(cache, indices):
    return torch.where(indices < 0, cache.pad_slot, indices).to(
        torch.int32
    ), indices.to(torch.int32)


def step(
    *,
    cache,
    scratch,
    state,
    slots,
    rows,
    counts,
    chunks,
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    out,
    nmax,
    num_k_heads=16,
    head_k_dim=128,
    eps=1e-6,
    kkt=(64, 4),
    bv=64,
):
    """Rows rows[i] .. rows[i] + counts[i] - 1 of chunk chunks[i] for request i. mixed_qkv / a / b / out hold
    nmax token rows per request (request i's row j at i * nmax + j; rows past counts[i] are ignored);
    counts[i] = 0 skips the request; rows + counts <= 64."""
    B = rows.shape[0]
    H = a.shape[1]
    Hg, K = num_k_heads, head_k_dim
    V = (mixed_qkv.shape[1] - 2 * Hg * K) // H
    ring_slots, state_slots = slots
    _close_pending_kernel[(B, H)](
        state,
        cache.k,
        cache.v,
        cache.g,
        cache.beta,
        cache.solved,
        cache.e_chunk,
        ring_slots,
        state_slots,
        rows,
        counts,
        chunks,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        num_warps=4,
    )
    _prep_kernel[(B, H)](
        mixed_qkv,
        mixed_qkv.stride(0),
        a,
        b,
        A_log,
        dt_bias,
        cache.k,
        cache.v,
        cache.g,
        cache.beta,
        scratch.q,
        scratch.g_cum,
        ring_slots,
        rows,
        counts,
        eps,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        NMAX=nmax,
        num_warps=8,
    )
    _solve_kernel[(B * H,)](
        cache.k,
        cache.beta,
        scratch.g_cum,
        cache.solved,
        scratch.cur,
        ring_slots,
        rows,
        counts,
        H=H,
        Hg=Hg,
        K=K,
        BK=kkt[0],
        num_warps=kkt[1],
    )
    _wu_kernel[(B * H,)](
        cache.k,
        cache.v,
        cache.beta,
        scratch.g_cum,
        scratch.cur,
        scratch.w,
        scratch.u,
        ring_slots,
        rows,
        counts,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        NMAX=nmax,
        num_warps=4,
    )
    _out_kernel[(V // bv, B * H)](
        state,
        cache.k,
        cache.v_new,
        scratch.g_cum,
        scratch.w,
        scratch.u,
        scratch.q,
        out,
        ring_slots,
        state_slots,
        rows,
        counts,
        K**-0.5,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BV=bv,
        NMAX=nmax,
        num_warps=4,
    )
    return out
