# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk_intra.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Stands in for the stock recurrent KDA decode (fused_recurrent_kda_packed_decode), which advances the FP32 state one
token at a time; that associates differently from the chunk-64 prefill, and no kernel configuration aligns the two.

Row-engine KDA decode on the TOP chunk-64 KDA program (``chunk_kda`` with the safe gate, the fused intra kernel and
the configs ``kda_top`` pins).

A decode step computes only its own row p of the open 64-token chunk, with the arithmetic ``chunk_kda`` applies to
that row in a whole-sequence prefill. Under TOP that program is prefix-stable: the fused intra kernel takes every
16-row block's gate reference at the block's first row, so row p of every stage depends only on rows <= p of its
chunk and on the FP32 state at the chunk start. (The unfused diagonal kernel references the block's middle row, or
its last row when the sequence ends inside it, which a decode cannot know: TOP never runs it.) The slot holds

- ``state[slot]``: the FP32 state E at the start of the open chunk (written once per chunk);
- rows 0..p-1 of the open chunk: l2-normed k, v, the raw gate, beta, v_new (the state kernel's per-row output) and
  the completed 16 x 16 FP32 blocks of the inverse (rows 0..47).

Per step, four launches with fixed grids (CUDA-graph capturable):
1. ``_prep_kernel`` (B, H): l2norm of q and k (l2norm_fwd's 16-row tile), row p and its raw gate and beta into the
   rings; then the prefill's own gate kernel over each slot's 64 ring rows (gate, chunk-local cumsum, beta).
2. ``_solve_kernel`` (B * H): row block J = p // 16 of Akk and Aqk (diagonal and off-diagonal), the diagonal
   block's forward substitution, the block merges and rows p of u and w (the fused intra kernel's arithmetic, its
   pinned BK and num_warps).
3. ``_out_kernel`` (V / BV, B * H): row p of v_new (state kernel) and of the output (output kernel); at row 63 the
   state advances over the chunk in place (state kernel arithmetic).

Under speculation a chain verify runs its root and drafts as consecutive one-row steps inside the root's chunk, and
the close is deferred: ``close_pending`` advances E over a chunk only once a step starts past it, so the rows a
verify writes past its accepted ones never reach E and are overwritten by the next step. A radix track that lands on
row 63 while the close is deferred takes ``close_to``: the same close, written to the track slot.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.chunk_intra import SOLVE_TRIL_DOT_PRECISION
from sglang.kernels.ops.attention.fla.op import exp2

CHUNK = tl.constexpr(64)
_CHUNK = 64
# Rows of 16-row blocks 0..2: the only completed blocks later rows of the chunk read.
SOLVED_ROWS = tl.constexpr(48)


@triton.jit
def _prep_kernel(
    qkv,
    stride_qkv,
    f_raw,
    stride_f,
    b_raw,
    stride_b,
    k_ring,
    v_ring,
    f_ring,
    braw_ring,
    q_out,
    slots,
    rows,
    eps,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    i_n, i_h = tl.program_id(0), tl.program_id(1)
    s = tl.load(slots + i_n).to(tl.int64)
    p = tl.load(rows + i_n)
    x = qkv + i_n.to(tl.int64) * stride_qkv
    if i_h * 16 < H:
        # l2norm_fwd_kernel's tile: 16 rows (16 heads of the token) x K; program i_h normalizes heads 16 i_h ...
        r16 = i_h * 16 + tl.arange(0, 16)
        m16 = r16 < H
        ck = tl.arange(0, K)
        for which in tl.static_range(2):
            b_x = tl.load(
                x + which * H * K + r16[:, None] * K + ck[None, :],
                mask=m16[:, None],
                other=0.0,
            ).to(tl.float32)
            b_var = tl.sum(b_x * b_x, axis=1)
            b_y = b_x / tl.sqrt(b_var + eps)[:, None]
            if which == 0:
                dst = q_out + (i_n * H + r16[:, None]) * K + ck[None, :]
            else:
                dst = k_ring + ((s * CHUNK + p) * H + r16[:, None]) * K + ck[None, :]
            tl.store(dst, b_y.to(tl.bfloat16), mask=m16[:, None])
    cv = tl.arange(0, V)
    tl.store(
        v_ring + ((s * CHUNK + p) * H + i_h) * V + cv,
        tl.load(x + 2 * H * K + i_h * V + cv),
    )
    ck = tl.arange(0, K)
    tl.store(
        f_ring + ((s * CHUNK + p) * H + i_h) * K + ck,
        tl.load(f_raw + i_n.to(tl.int64) * stride_f + i_h * K + ck),
    )
    tl.store(
        braw_ring + (s * CHUNK + p) * H + i_h,
        tl.load(b_raw + i_n.to(tl.int64) * stride_b + i_h),
    )


@triton.jit
def _blk(ptr, rb, cb, ld: tl.constexpr):
    r = rb * 16 + tl.arange(0, 16)
    c = cb * 16 + tl.arange(0, 16)
    return tl.load(ptr + r[:, None] * ld + c[None, :])


@triton.jit
def _put(ptr, rb, cb, x, ld: tl.constexpr):
    r = rb * 16 + tl.arange(0, 16)
    c = cb * 16 + tl.arange(0, 16)
    tl.store(ptr + r[:, None] * ld + c[None, :], x)


@triton.jit
def _mm(a, b):
    return tl.dot(a, b, input_precision=SOLVE_TRIL_DOT_PRECISION)


@triton.jit
def _kgt(k_rows, g_rows, j, b_gn, o_k, HK: tl.constexpr):
    """[BK, 16] bf16(k_j * exp2(gn - g_j))^T of earlier block j, as the intra kernel's off-diagonal dots take it."""
    rj = j * 16 + tl.arange(0, 16)
    b_kj = tl.load(k_rows + rj[:, None] * HK + o_k[None, :]).to(tl.float32)
    b_gj = tl.load(g_rows + rj[:, None] * HK + o_k[None, :])
    return tl.trans(b_kj * exp2(b_gn[None, :] - b_gj)).to(tl.bfloat16)


@triton.jit
def _solve_kernel(
    k_ring,
    v_ring,
    beta_ring,
    g_cum,
    q_in,
    solved,
    aqk_row,
    w_row,
    u_row,
    slots,
    rows,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Rows of block J = p // 16 of the inverse (cached into ``solved`` when block J <= 2 completes), row p of Aqk
    (bf16, scaled; zero past p) into aqk_row[n, h], rows p of u and w into u_row, w_row."""
    i_n, i_h = tl.program_id(0) // H, tl.program_id(0) % H
    s = tl.load(slots + i_n).to(tl.int64)
    p = tl.load(rows + i_n)
    J = p // 16
    HK = H * K
    k_rows = k_ring + (s * CHUNK * H + i_h) * K
    g_rows = g_cum + (s * CHUNK * H + i_h) * K
    o_i = tl.arange(0, 16)
    rJ = J * 16 + o_i
    m_J = rJ <= p
    is_p = rJ == p

    b_AqD = tl.zeros([16, 16], dtype=tl.float32)
    b_AkD = tl.zeros([16, 16], dtype=tl.float32)
    b_Aq0 = tl.zeros([16, 16], dtype=tl.float32)
    b_Ak0 = tl.zeros([16, 16], dtype=tl.float32)
    b_Aq1 = tl.zeros([16, 16], dtype=tl.float32)
    b_Ak1 = tl.zeros([16, 16], dtype=tl.float32)
    b_Aq2 = tl.zeros([16, 16], dtype=tl.float32)
    b_Ak2 = tl.zeros([16, 16], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_qJ = tl.where(
            is_p[:, None],
            tl.load(q_in + (i_n * H + i_h) * K + o_k).to(tl.float32)[None, :],
            0.0,
        )
        b_kJ = tl.load(
            k_rows + rJ[:, None] * HK + o_k[None, :], mask=m_J[:, None], other=0.0
        ).to(tl.float32)
        b_gJ = tl.load(
            g_rows + rJ[:, None] * HK + o_k[None, :], mask=m_J[:, None], other=0.0
        )
        b_gn = tl.load(g_rows + (J * 16) * HK + o_k, mask=m_k, other=0)
        if J == 0:
            b_gm = tl.clamp(b_gJ - b_gn[None, :], -126.0, 126.0)
            b_gq = tl.where(m_J[:, None], exp2(b_gm), 0.0)
            b_gkd = tl.where(m_J[:, None], exp2(-b_gm), 0.0)
        else:
            b_gq = tl.where(m_J[:, None], exp2(b_gJ - b_gn[None, :]), 0)
            b_gmd = tl.clamp(b_gn[None, :] - b_gJ, -126.0, 126.0)
            b_gkd = tl.where(m_J[:, None], exp2(b_gmd), 0.0)
        b_kgt_d = tl.trans(b_kJ * b_gkd)
        b_AqD += tl.dot(b_qJ * b_gq, b_kgt_d)
        b_AkD += tl.dot(b_kJ * b_gq, b_kgt_d)
        if J >= 1:
            b_qg = (b_qJ * b_gq).to(tl.bfloat16)
            b_kg = (b_kJ * b_gq).to(tl.bfloat16)
            b_kgt = _kgt(k_rows, g_rows, 0, b_gn, o_k, HK)
            b_Aq0 += tl.dot(b_qg, b_kgt)
            b_Ak0 += tl.dot(b_kg, b_kgt)
            if J >= 2:
                b_kgt = _kgt(k_rows, g_rows, 1, b_gn, o_k, HK)
                b_Aq1 += tl.dot(b_qg, b_kgt)
                b_Ak1 += tl.dot(b_kg, b_kgt)
                if J >= 3:
                    b_kgt = _kgt(k_rows, g_rows, 2, b_gn, o_k, HK)
                    b_Aq2 += tl.dot(b_qg, b_kgt)
                    b_Ak2 += tl.dot(b_kg, b_kgt)

    b_bJ = tl.load(beta_ring + (s * CHUNK + rJ) * H + i_h, mask=m_J, other=0.0)
    # Row p of Aqk (scaled, bf16 as the intra kernel stores it); columns past the diagonal block stay zero.
    aq = aqk_row + (i_n * H + i_h) * CHUNK
    c64 = tl.arange(0, CHUNK)
    tl.store(aq + c64, tl.zeros([CHUNK], dtype=tl.bfloat16))
    tl.debug_barrier()
    m_Aqk_diag = o_i[:, None] >= o_i[None, :]
    m_Akk_diag = o_i[:, None] > o_i[None, :]
    b_AqD = tl.where(m_Aqk_diag, b_AqD, 0.0)
    b_AkD = tl.where(m_Akk_diag, b_AkD, 0.0)
    tl.store(
        aq + J * 16 + o_i[None, :] + 0 * o_i[:, None],
        (b_AqD * scale).to(tl.bfloat16),
        mask=is_p[:, None],
    )
    if J >= 1:
        tl.store(
            aq + o_i[None, :] + 0 * o_i[:, None],
            (b_Aq0 * scale).to(tl.bfloat16),
            mask=is_p[:, None],
        )
        b_Ak0 = b_Ak0 * b_bJ[:, None]
    if J >= 2:
        tl.store(
            aq + 16 + o_i[None, :] + 0 * o_i[:, None],
            (b_Aq1 * scale).to(tl.bfloat16),
            mask=is_p[:, None],
        )
        b_Ak1 = b_Ak1 * b_bJ[:, None]
    if J >= 3:
        tl.store(
            aq + 32 + o_i[None, :] + 0 * o_i[:, None],
            (b_Aq2 * scale).to(tl.bfloat16),
            mask=is_p[:, None],
        )
        b_Ak2 = b_Ak2 * b_bJ[:, None]
    b_AkD = b_AkD * b_bJ[:, None]

    # Forward substitution on the diagonal block (rows past p are never read by rows <= p).
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    b_Ai = -tl.where(m_A, b_AkD, 0)
    for i in range(2, p % 16 + 1):
        b_a = -tl.sum(tl.where((o_i == i)[:, None], b_AkD, 0.0), 0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I

    sv = solved + (s * H + i_h) * SOLVED_ROWS * SOLVED_ROWS
    done = (p % 16 == 15) & (J <= 2)
    if done:
        _put(sv, J, J, b_Ai, SOLVED_ROWS)
    # Merged inverse blocks of row block J (the intra kernel's formulas; earlier blocks from the solved cache).
    b_X0 = tl.zeros([16, 16], dtype=tl.float32)
    b_X1 = tl.zeros([16, 16], dtype=tl.float32)
    b_X2 = tl.zeros([16, 16], dtype=tl.float32)
    if J == 1:
        b_Ai00 = _blk(sv, 0, 0, SOLVED_ROWS)
        b_X0 = -_mm(_mm(b_Ai, b_Ak0), b_Ai00)
        if done:
            _put(sv, 1, 0, b_X0, SOLVED_ROWS)
    elif J == 2:
        b_Ai00 = _blk(sv, 0, 0, SOLVED_ROWS)
        b_Ai10 = _blk(sv, 1, 0, SOLVED_ROWS)
        b_Ai11 = _blk(sv, 1, 1, SOLVED_ROWS)
        b_X1 = -_mm(_mm(b_Ai, b_Ak1), b_Ai11)
        b_X0 = -_mm(b_Ai, _mm(b_Ak0, b_Ai00) + _mm(b_Ak1, b_Ai10))
        if done:
            _put(sv, 2, 1, b_X1, SOLVED_ROWS)
            _put(sv, 2, 0, b_X0, SOLVED_ROWS)
    elif J == 3:
        b_Ai00 = _blk(sv, 0, 0, SOLVED_ROWS)
        b_Ai10 = _blk(sv, 1, 0, SOLVED_ROWS)
        b_Ai11 = _blk(sv, 1, 1, SOLVED_ROWS)
        b_Ai20 = _blk(sv, 2, 0, SOLVED_ROWS)
        b_Ai21 = _blk(sv, 2, 1, SOLVED_ROWS)
        b_Ai22 = _blk(sv, 2, 2, SOLVED_ROWS)
        b_X2 = -_mm(_mm(b_Ai, b_Ak2), b_Ai22)
        b_X1 = -_mm(b_Ai, _mm(b_Ak1, b_Ai11) + _mm(b_Ak2, b_Ai21))
        b_X0 = -_mm(b_Ai, _mm(b_Ak0, b_Ai00) + _mm(b_Ak1, b_Ai10) + _mm(b_Ak2, b_Ai20))

    _recompute(
        b_Ai,
        b_X0,
        b_X1,
        b_X2,
        J,
        p,
        s,
        i_n,
        i_h,
        k_rows,
        v_ring,
        beta_ring,
        g_rows,
        w_row,
        u_row,
        H,
        K,
        V,
        BK,
        BV,
    )


@triton.jit
def _recompute(
    b_AiJJ,
    b_X0,
    b_X1,
    b_X2,
    J,
    p,
    s,
    i_n,
    i_h,
    k_rows,
    v_ring,
    beta_ring,
    g_rows,
    w_row,
    u_row,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Rows p of u = A^-1 bf16(v * beta) and w = A^-1 bf16(k * beta * exp2(gk)), with the A^-1 blocks in bf16 and
    the fused recompute's per-block dots summed left to right."""
    HK = H * K
    o_i = tl.arange(0, 16)
    rJ = J * 16 + o_i
    m_J = rJ <= p
    is_p = rJ == p
    aJJ = b_AiJJ.to(tl.bfloat16)
    x0 = b_X0.to(tl.bfloat16)
    x1 = b_X1.to(tl.bfloat16)
    x2 = b_X2.to(tl.bfloat16)
    r0 = o_i
    r1 = 16 + o_i
    r2 = 32 + o_i
    b_b0 = tl.load(beta_ring + (s * CHUNK + r0) * H + i_h)
    b_b1 = tl.load(beta_ring + (s * CHUNK + r1) * H + i_h)
    b_b2 = tl.load(beta_ring + (s * CHUNK + r2) * H + i_h)
    b_bJ = tl.load(beta_ring + (s * CHUNK + rJ) * H + i_h, mask=m_J, other=0.0)
    v_rows = v_ring + (s * CHUNK * H + i_h) * V
    for i_v in range(tl.cdiv(V, BV)):
        cv = i_v * BV + tl.arange(0, BV)
        b_vJ = tl.load(
            v_rows + rJ[:, None] * (H * V) + cv[None, :], mask=m_J[:, None], other=0.0
        )
        b_vbJ = (b_vJ * b_bJ[:, None]).to(b_vJ.dtype)
        if J == 0:
            b_u = tl.dot(aJJ, b_vbJ)
        else:
            b_v0 = tl.load(v_rows + r0[:, None] * (H * V) + cv[None, :])
            b_vb0 = (b_v0 * b_b0[:, None]).to(b_v0.dtype)
            if J == 1:
                b_u = tl.dot(x0, b_vb0) + tl.dot(aJJ, b_vbJ)
            else:
                b_v1 = tl.load(v_rows + r1[:, None] * (H * V) + cv[None, :])
                b_vb1 = (b_v1 * b_b1[:, None]).to(b_v1.dtype)
                if J == 2:
                    b_u = tl.dot(x0, b_vb0) + tl.dot(x1, b_vb1) + tl.dot(aJJ, b_vbJ)
                else:
                    b_v2 = tl.load(v_rows + r2[:, None] * (H * V) + cv[None, :])
                    b_vb2 = (b_v2 * b_b2[:, None]).to(b_v2.dtype)
                    b_u = (
                        tl.dot(x0, b_vb0)
                        + tl.dot(x1, b_vb1)
                        + tl.dot(x2, b_vb2)
                        + tl.dot(aJJ, b_vbJ)
                    )
        tl.store(
            u_row + (i_n * H + i_h) * V + cv[None, :] + 0 * rJ[:, None],
            b_u.to(tl.bfloat16),
            mask=is_p[:, None],
        )
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        b_kJ = tl.load(
            k_rows + rJ[:, None] * HK + o_k[None, :], mask=m_J[:, None], other=0.0
        )
        b_gJ = tl.load(
            g_rows + rJ[:, None] * HK + o_k[None, :], mask=m_J[:, None], other=0.0
        )
        b_kbJ = (b_kJ * b_bJ[:, None] * exp2(b_gJ)).to(b_kJ.dtype)
        if J == 0:
            b_w = tl.dot(aJJ, b_kbJ)
        else:
            b_k0 = tl.load(k_rows + r0[:, None] * HK + o_k[None, :])
            b_g0 = tl.load(g_rows + r0[:, None] * HK + o_k[None, :])
            b_kb0 = (b_k0 * b_b0[:, None] * exp2(b_g0)).to(b_k0.dtype)
            if J == 1:
                b_w = tl.dot(x0, b_kb0) + tl.dot(aJJ, b_kbJ)
            else:
                b_k1 = tl.load(k_rows + r1[:, None] * HK + o_k[None, :])
                b_g1 = tl.load(g_rows + r1[:, None] * HK + o_k[None, :])
                b_kb1 = (b_k1 * b_b1[:, None] * exp2(b_g1)).to(b_k1.dtype)
                if J == 2:
                    b_w = tl.dot(x0, b_kb0) + tl.dot(x1, b_kb1) + tl.dot(aJJ, b_kbJ)
                else:
                    b_k2 = tl.load(k_rows + r2[:, None] * HK + o_k[None, :])
                    b_g2 = tl.load(g_rows + r2[:, None] * HK + o_k[None, :])
                    b_kb2 = (b_k2 * b_b2[:, None] * exp2(b_g2)).to(b_k2.dtype)
                    b_w = (
                        tl.dot(x0, b_kb0)
                        + tl.dot(x1, b_kb1)
                        + tl.dot(x2, b_kb2)
                        + tl.dot(aJJ, b_kbJ)
                    )
        tl.store(
            w_row + (i_n * H + i_h) * K + o_k[None, :] + 0 * rJ[:, None],
            b_w.to(tl.bfloat16),
            mask=is_p[:, None],
        )


@triton.jit
def _out_kernel(
    state,
    k_ring,
    vnew_ring,
    g_cum,
    aqk_row,
    w_row,
    u_row,
    q_in,
    out,
    stride_out,
    slots,
    state_slots,
    rows,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    ROWS: tl.constexpr,
    CLOSE: tl.constexpr,
):
    """Row p of v_new (state kernel) and of the output (output kernel), on a ROWS-row tile holding row p (the other
    rows zero); with CLOSE, at row 63 the state kernel's step over the chunk, in place."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    HK = H * K
    s = tl.load(slots + i_n).to(tl.int64)
    ss = tl.maximum(tl.load(state_slots + i_n), 0).to(tl.int64)
    p = tl.load(rows + i_n)
    c64 = tl.arange(0, 64)
    r64 = tl.arange(0, CHUNK)
    cv = i_v * BV + tl.arange(0, BV)
    rt = tl.arange(0, ROWS)
    if ROWS == 16:
        r_tile = p // 16 * 16 + rt
    else:
        r_tile = rt
    is_p = (r_tile == p)[:, None]
    k_rows = k_ring + (s * CHUNK * H + i_h) * K
    g_rows = g_cum + (s * CHUNK * H + i_h) * K
    h0 = state + (ss * H + i_h) * V * K

    # State kernel: v_new = u - (w[:, :64] @ bf16(E[:, :64])^T + w[:, 64:] @ bf16(E[:, 64:])^T).
    w_p = w_row + (i_n * H + i_h) * K
    b_w1 = tl.where(is_p, tl.load(w_p + c64)[None, :], 0.0).to(tl.bfloat16)
    b_w2 = tl.where(is_p, tl.load(w_p + 64 + c64)[None, :], 0.0).to(tl.bfloat16)
    b_h1 = tl.load(h0 + cv[:, None] * K + c64[None, :])
    b_h2 = tl.load(h0 + cv[:, None] * K + 64 + c64[None, :])
    b_vn = tl.dot(b_w1, tl.trans(b_h1).to(tl.bfloat16))
    b_vn += tl.dot(b_w2, tl.trans(b_h2).to(tl.bfloat16))
    b_u = tl.where(is_p, tl.load(u_row + (i_n * H + i_h) * V + cv)[None, :], 0.0).to(
        tl.bfloat16
    )
    b_vn = b_u - b_vn
    vn_rows = vnew_ring + (s * CHUNK * H + i_h) * V
    tl.store(
        vn_rows + r_tile[:, None] * (H * V) + cv[None, :],
        b_vn.to(tl.bfloat16),
        mask=is_p,
    )
    tl.debug_barrier()

    # Output kernel: bf16(bf16(q * scale) * exp2(g)) @ bf16(E)^T over BK blocks, plus the causal intra-chunk term.
    b_o = tl.zeros([ROWS, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        b_q = tl.where(
            is_p, tl.load(q_in + (i_n * H + i_h) * K + o_k)[None, :], 0.0
        ).to(tl.bfloat16)
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(
            g_rows + r_tile[:, None] * HK + o_k[None, :], mask=is_p, other=0.0
        )
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(h0 + cv[:, None] * K + o_k[None, :]).to(tl.bfloat16)
        # The output kernel's own guard; kept so the accumulation compiles as there.
        if i_k >= 0:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    b_vt = tl.load(
        vn_rows + r64[:, None] * (H * V) + cv[None, :],
        mask=(r64 <= p)[:, None],
        other=0.0,
    )
    b_A = tl.where(
        is_p, tl.load(aqk_row + (i_n * H + i_h) * CHUNK + c64)[None, :], 0.0
    ).to(b_vt.dtype)
    b_o += tl.dot(b_A, b_vt, allow_tf32=False)
    tl.store(
        out
        + i_n.to(tl.int64) * stride_out
        + i_h * V
        + cv[None, :]
        + r_tile[:, None] * 0,
        b_o.to(tl.bfloat16),
        mask=is_p,
    )

    if CLOSE:
        if (p == CHUNK - 1) & (tl.load(state_slots + i_n) >= 0):
            _close_chunk(h0, h0, k_rows, g_rows, vn_rows, cv, H, K, V)


@triton.jit
def _close_chunk(
    h0,
    h_out,
    k_rows,
    g_rows,
    vn_rows,
    cv,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    """The chunk is complete: E' = E * exp2(g_last) + (kg^T @ bf16(v_new))^T per 64-wide K block, with
    kg = bf16(k * exp2(g_last - g)) as the fused recompute writes it; E read at ``h0``, E' written at ``h_out`` (in
    place when they are equal), on the V tile ``cv``."""
    HK = H * K
    c64 = tl.arange(0, 64)
    r64 = tl.arange(0, CHUNK)
    b_vc = tl.load(vn_rows + r64[:, None] * (H * V) + cv[None, :])
    for i_kb in tl.static_range(K // 64):
        ck = i_kb * 64 + c64
        b_gl = tl.load(g_rows + (CHUNK - 1) * HK + ck)
        b_kr = tl.load(k_rows + r64[:, None] * HK + ck[None, :])
        b_gk = tl.load(g_rows + r64[:, None] * HK + ck[None, :])
        b_kg = (b_kr * exp2(b_gl[None, :] - b_gk)).to(tl.bfloat16)
        b_E = tl.load(h0 + cv[:, None] * K + ck[None, :])
        b_new = tl.fma(b_E, exp2(b_gl)[None, :], tl.trans(tl.dot(tl.trans(b_kg), b_vc)))
        tl.store(h_out + cv[:, None] * K + ck[None, :], b_new)


@triton.jit
def _close_kernel(
    state,
    k_ring,
    vnew_ring,
    g_cum,
    e_chunk,
    slots,
    state_slots,
    chunks,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """The deferred close: for a request whose step starts in a later chunk than E's, the out kernel's step over the
    chunk E opens (all 64 rows cached), in place."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    s = tl.load(slots + i_n).to(tl.int64)
    ss = tl.load(state_slots + i_n)
    if (ss >= 0) & (tl.load(chunks + i_n) > tl.load(e_chunk + s)):
        cv = i_v * BV + tl.arange(0, BV)
        h0 = state + (ss.to(tl.int64) * H + i_h) * V * K
        _close_chunk(
            h0,
            h0,
            k_ring + (s * CHUNK * H + i_h) * K,
            g_cum + (s * CHUNK * H + i_h) * K,
            vnew_ring + (s * CHUNK * H + i_h) * V,
            cv,
            H,
            K,
            V,
        )


@triton.jit
def _close_to_kernel(
    state,
    k_ring,
    vnew_ring,
    g_cum,
    slots,
    state_slots,
    dst_slots,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """The close of the chunk E opens (all 64 rows cached), written to ``dst_slots`` instead of in place."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    s = tl.load(slots + i_n).to(tl.int64)
    ss = tl.load(state_slots + i_n)
    dst = tl.load(dst_slots + i_n)
    if (ss >= 0) & (dst >= 0):
        cv = i_v * BV + tl.arange(0, BV)
        _close_chunk(
            state + (ss.to(tl.int64) * H + i_h) * V * K,
            state + (dst.to(tl.int64) * H + i_h) * V * K,
            k_ring + (s * CHUNK * H + i_h) * K,
            g_cum + (s * CHUNK * H + i_h) * K,
            vnew_ring + (s * CHUNK * H + i_h) * V,
            cv,
            H,
            K,
            V,
        )


class KDARowCache:
    """Per-layer rows of each slot's open chunk. The last slot absorbs padded rows (slot -1)."""

    def __init__(self, *, num_slots, num_heads, head_dim, device):
        assert head_dim == 128
        n, H, K = num_slots + 1, num_heads, head_dim
        self.pad_slot = num_slots
        self.k = torch.zeros(n, _CHUNK, H, K, dtype=torch.bfloat16, device=device)
        self.v = torch.zeros(n, _CHUNK, H, K, dtype=torch.bfloat16, device=device)
        self.v_new = torch.zeros(n, _CHUNK, H, K, dtype=torch.bfloat16, device=device)
        self.f = torch.zeros(n, _CHUNK, H, K, dtype=torch.bfloat16, device=device)
        self.b = torch.zeros(n, _CHUNK, H, dtype=torch.bfloat16, device=device)
        self.beta = torch.zeros(n, _CHUNK, H, dtype=torch.float32, device=device)
        self.solved = torch.zeros(
            n, H, SOLVED_ROWS, SOLVED_ROWS, dtype=torch.float32, device=device
        )
        # The chunk E stands at, read by the deferred close (``close_pending``); unused when decode closes in place.
        self.e_chunk = torch.zeros(n, dtype=torch.int32, device=device)

    def bytes_per_slot(self):
        return sum(
            t[0].numel() * t.element_size()
            for t in (
                self.k,
                self.v,
                self.v_new,
                self.f,
                self.b,
                self.beta,
                self.solved,
                self.e_chunk,
            )
        )


class KDAScratch:
    """Buffers of one step, shared by all layers: per batch row, and the gate cumsum per slot (the prefill's gate
    kernel writes it at the slot's ring rows)."""

    def __init__(self, *, max_bs, num_slots, H, K, device):
        self.q = torch.empty(max_bs, H, K, dtype=torch.bfloat16, device=device)
        self.aqk = torch.empty(max_bs, H, _CHUNK, dtype=torch.bfloat16, device=device)
        self.w = torch.empty(max_bs, H, K, dtype=torch.bfloat16, device=device)
        self.u = torch.empty(max_bs, H, K, dtype=torch.bfloat16, device=device)
        self.g_cum = torch.empty(
            num_slots + 1, _CHUNK, H, K, dtype=torch.float32, device=device
        )


def _gate_cumsum(*, cache, g_cum, ring_slots, A_log, dt_bias, lower_bound, gate_scale):
    """The prefill's own gate kernel over each request's 64 ring rows (a varlen call whose sequence i is slot
    s_i's rows): the cumsum and beta of rows <= p are those of any prefill reaching row p, whatever the rows past p
    hold."""
    from sglang.kernels.ops.attention.fla.kda import kda_gate_chunk_cumsum_vector_kernel

    B = ring_slots.numel()
    _, _, H, K = cache.f.shape
    starts = ring_slots.to(torch.int32) * _CHUNK
    seq_cu = torch.stack((starts, starts + _CHUNK), dim=1).flatten()
    chunk_idx = torch.stack(
        (
            torch.arange(0, 2 * B, 2, dtype=torch.int32, device=starts.device),
            torch.zeros_like(starts),
        ),
        dim=1,
    ).flatten()
    braw = cache.b.view(-1, H)
    kda_gate_chunk_cumsum_vector_kernel[
        lambda meta: (triton.cdiv(K, meta["BS"]), B, H)
    ](
        s=cache.f.view(-1, H, K),
        A_log=A_log,
        dt_bias=dt_bias,
        o=g_cum,
        scale=gate_scale,
        cu_seqlens=seq_cu,
        chunk_indices=chunk_idx,
        lower_bound=lower_bound,
        beta=braw,
        beta_out=cache.beta,
        beta_stride_b=0,
        beta_stride_t=braw.stride(0),
        beta_stride_h=braw.stride(1),
        T=_CHUNK,
        H=H,
        S=K,
        BT=_CHUNK,
    )


def decode(
    *,
    cache,
    scratch,
    state,
    slots,
    rows,
    mixed_qkv,
    f,
    b,
    A_log,
    dt_bias,
    lower_bound,
    out,
    config,
    close=True,
    eps=1e-6,
    rows_tile=16,
):
    """One token per request. mixed_qkv: [B, 3 * H * K] conv output (q, k before l2norm, then v); f: [B, H * K] raw
    gate; b: [B, H] raw beta (each row-strided, unit stride within a row); rows: [B] int32 position in the 64-token
    chunk; slots: (cache slots, state slots); state: [slots, H, V, K] FP32 boundary states; config:
    kda_top.TopKDAConfig; out: [B, H, V], row-strided. With ``close`` a row 63 advances E in place; without it E
    stays at the chunk start until ``close_pending`` runs on the next chunk's first step."""
    B, H = b.shape
    K = 128
    V = K
    ring_slots, state_slots = slots
    _prep_kernel[(B, H)](
        mixed_qkv,
        mixed_qkv.stride(0),
        f,
        f.stride(0),
        b,
        b.stride(0),
        cache.k,
        cache.v,
        cache.f,
        cache.b,
        scratch.q,
        ring_slots,
        rows,
        eps,
        H=H,
        K=K,
        V=V,
        num_warps=8,
    )
    _gate_cumsum(
        cache=cache,
        g_cum=scratch.g_cum,
        ring_slots=ring_slots,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        gate_scale=config.gate_scale,
    )
    _solve_kernel[(B * H,)](
        cache.k,
        cache.v,
        cache.beta,
        scratch.g_cum,
        scratch.q,
        cache.solved,
        scratch.aqk,
        scratch.w,
        scratch.u,
        ring_slots,
        rows,
        K**-0.5,
        H=H,
        K=K,
        V=V,
        BK=config.intra_bk,
        BV=config.intra_bv,
        num_warps=config.intra_warps,
    )
    _out_kernel[(V // config.out_bv, B * H)](
        state,
        cache.k,
        cache.v_new,
        scratch.g_cum,
        scratch.aqk,
        scratch.w,
        scratch.u,
        scratch.q,
        out,
        out.stride(0),
        ring_slots,
        state_slots,
        rows,
        K**-0.5,
        H=H,
        K=K,
        V=V,
        BV=config.out_bv,
        BK=config.out_bk,
        ROWS=rows_tile,
        CLOSE=close,
        num_warps=config.out_warps,
    )
    return out


def close_pending(
    *, cache, scratch, state, slots, chunks, A_log, dt_bias, lower_bound, config
):
    """Before a step that decodes without ``close``: for each request whose step starts in chunk ``chunks`` past the
    chunk E stands at, E advances over that chunk's 64 cached rows (the in-place close's arithmetic), and
    ``cache.e_chunk`` follows. chunks: [B] int32 chunk index of each request's first row this step; slots as
    ``decode``. A verify never closes a chunk its drafts reach, so rejected rows never enter E."""
    B = chunks.numel()
    H = cache.b.shape[2]
    K = V = 128
    ring_slots, state_slots = slots
    _gate_cumsum(
        cache=cache,
        g_cum=scratch.g_cum,
        ring_slots=ring_slots,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        gate_scale=config.gate_scale,
    )
    _close_kernel[(V // config.out_bv, B * H)](
        state,
        cache.k,
        cache.v_new,
        scratch.g_cum,
        cache.e_chunk,
        ring_slots,
        state_slots,
        chunks,
        H=H,
        K=K,
        V=V,
        BV=config.out_bv,
        num_warps=config.out_warps,
    )
    e_old = cache.e_chunk[ring_slots]
    cache.e_chunk[ring_slots] = torch.where(
        (state_slots >= 0) & (chunks > e_old), chunks, e_old
    )


def close_to(
    *, cache, scratch, state, slots, dst_slots, A_log, dt_bias, lower_bound, config
):
    """For rows with ``dst_slots`` >= 0 whose open chunk has all 64 rows cached and its close still deferred: the
    chunk's end state (the in-place close's arithmetic) into ``state[dst_slots]``; E and ``cache.e_chunk`` stay.
    slots as ``decode``; dst_slots: [B] int32, -1 for none."""
    B = dst_slots.numel()
    H = cache.b.shape[2]
    K = V = 128
    ring_slots, state_slots = slots
    _gate_cumsum(
        cache=cache,
        g_cum=scratch.g_cum,
        ring_slots=ring_slots,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        gate_scale=config.gate_scale,
    )
    _close_to_kernel[(V // config.out_bv, B * H)](
        state,
        cache.k,
        cache.v_new,
        scratch.g_cum,
        ring_slots,
        state_slots,
        dst_slots,
        H=H,
        K=K,
        V=V,
        BV=config.out_bv,
        num_warps=config.out_warps,
    )
