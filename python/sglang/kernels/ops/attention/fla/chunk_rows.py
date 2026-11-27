# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Stands in for the stock recurrent GDN decode kernels (fused_recurrent, fused_sigmoid_gating_delta_rule_update),
which advance the FP32 state one token at a time; that associates differently from the chunk-64 prefill, and no
kernel configuration aligns the two.

TOP chunked-GDN row engine: decode on FLA's chunk-64 program.

A decode step computes only its own row p of the open 64-token chunk, with the arithmetic `chunk_gated_delta_rule`
(fixed 64-row output tile, FP32 state) applies to that row in a whole-sequence prefill. FLA's chunk program is
prefix-stable: row p of every stage depends only on rows <= p of its chunk and on the FP32 state at the chunk
start, so the slot holds

- `state[slot]`: the FP32 state E at the start of the open chunk (the boundary state, written once per chunk);
- rows 0..p-1 of the open chunk: l2-normalized k (per key head), v, raw gate g and beta (the inputs), v_new
  (the state kernel's per-row output, read by the output of later rows) and the completed 16x16 FP32 blocks of
  (I + A)^{-1} (read by the block merges of later rows).

Per step, three launches with fixed grids (CUDA-graph capturable):
1. `_prep_kernel` (B, H): l2norm of q, k on l2norm_fwd's 16-row tile, gating (fused_gdn_gating arithmetic), row p
   into the rings, chunk-local cumsum (chunk_local_cumsum arithmetic).
2. `_solve_kernel` (B * H): rows of the current 16-row block of beta * K K^T * gate, its forward substitution and
   the block merges (chunk_gated_delta_rule_fwd_kkt_solve_kernel arithmetic, same BK and num_warps), from the cached
   blocks of earlier 16-row blocks.
3. `_out_kernel` (V / BV, B * H): rows p of w and u (recompute_w_u arithmetic), of v_new (state kernel) and of the
   output (output kernel); at row 63 the boundary state advances over the chunk (state kernel arithmetic, in place).
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.chunk_fwd import (
    _MERGE_DOT_PRECISION,
    chunk_gated_delta_rule_fwd_kkt_solve_kernel,
)
from sglang.kernels.ops.attention.fla.op import exp, safe_exp

CHUNK = tl.constexpr(64)
# Rows of 16-row blocks 0..2: the only completed blocks later rows of the chunk read.
SOLVED_ROWS = tl.constexpr(48)


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
    eps,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    i_n, i_h = tl.program_id(0), tl.program_id(1)
    s = tl.load(slots + i_n).to(tl.int64)
    p = tl.load(rows + i_n)
    x = qkv + i_n.to(tl.int64) * stride_qkv
    if i_h == 0:
        # l2norm_fwd_kernel's tile: 16 rows (here one token's Hg = 16 key heads) x K, 8 warps.
        r16 = tl.arange(0, 16)
        ck = tl.arange(0, K)
        for which in tl.static_range(2):
            b_x = tl.load(x + which * Hg * K + r16[:, None] * K + ck[None, :]).to(
                tl.float32
            )
            b_var = tl.sum(b_x * b_x, axis=1)
            b_y = b_x / tl.sqrt(b_var + eps)[:, None]
            if which == 0:
                dst = q_out + (i_n * Hg + r16[:, None]) * K + ck[None, :]
            else:
                dst = k_ring + ((s * CHUNK + p) * Hg + r16[:, None]) * K + ck[None, :]
            tl.store(dst, b_y.to(tl.bfloat16))
    cv = tl.arange(0, V)
    tl.store(
        v_ring + ((s * CHUNK + p) * H + i_h) * V + cv,
        tl.load(x + 2 * Hg * K + i_h * V + cv),
    )
    # fused_gdn_gating_kernel (beta = 1, threshold = 20); beta is rounded to b's dtype as there.
    blk_x = tl.load(a + i_n * H + i_h).to(tl.float32) + tl.load(dt_bias + i_h).to(
        tl.float32
    )
    softplus_x = tl.where(
        1.0 * blk_x <= 20.0, (1 / 1.0) * tl.log(1 + tl.exp(1.0 * blk_x)), blk_x
    )
    blk_g = -tl.exp(tl.load(A_log + i_h).to(tl.float32)) * softplus_x
    blk_b = tl.load(b + i_n * H + i_h)
    blk_beta = tl.sigmoid(blk_b.to(tl.float32)).to(blk_b.dtype).to(tl.float32)
    tl.store(g_ring + (s * CHUNK + p) * H + i_h, blk_g)
    tl.store(beta_ring + (s * CHUNK + p) * H + i_h, blk_beta)
    # chunk_local_cumsum_scalar_kernel over the partial chunk (rows past p are zero, as in a call ending at p).
    r64 = tl.arange(0, CHUNK)
    b_s = tl.load(g_ring + (s * CHUNK + r64) * H + i_h, mask=r64 < p, other=0.0).to(
        tl.float32
    )
    b_s = tl.where(r64 == p, blk_g, b_s)
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(g_cum + (i_n * CHUNK + r64) * H + i_h, b_o)


@triton.jit
def _kk(k_rows, j, m, p, Hg: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    """[16, 16] block (j, m) of K K^T as the fused kkt kernel accumulates it (rows past p are zero)."""
    rj = j * 16 + tl.arange(0, 16)
    rm = m * 16 + tl.arange(0, 16)
    acc = tl.zeros([16, 16], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        ck = i_k * BK + tl.arange(0, BK)
        b_kj = tl.load(
            k_rows + rj[:, None] * (Hg * K) + ck[None, :],
            mask=(rj <= p)[:, None],
            other=0.0,
        )
        b_km = tl.load(
            k_rows + rm[:, None] * (Hg * K) + ck[None, :],
            mask=(rm <= p)[:, None],
            other=0.0,
        )
        acc += tl.dot(b_kj, tl.trans(b_km))
    return acc


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
    return tl.dot(a, b, input_precision=_MERGE_DOT_PRECISION)


@triton.jit
def _solve_kernel(
    k_ring,
    beta_ring,
    g_cum,
    solved,
    cur,
    slots,
    rows,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    SLD: tl.constexpr,
):
    """Rows of 16-row block J = p // 16 of (I + A)^{-1} into cur[n, h] ([16, 64] FP32, columns past the block are
    zero); when the block completes (J <= 2), also into the slot's solved cache."""
    i_n, i_h = tl.program_id(0) // H, tl.program_id(0) % H
    s = tl.load(slots + i_n).to(tl.int64)
    p = tl.load(rows + i_n)
    J = p // 16
    k_rows = k_ring + (s * CHUNK * Hg + i_h // (H // Hg)) * K
    o_i = tl.arange(0, 16)
    rJ = J * 16 + o_i
    g_row = g_cum + i_n * CHUNK * H + i_h
    b_gJ = tl.load(g_row + rJ * H)
    b_bJ = tl.load(beta_ring + (s * CHUNK + rJ) * H + i_h, mask=rJ <= p, other=0.0)
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    m_J = rJ <= p

    b_A = _kk(k_rows, J, J, p, Hg, K, BK)
    b_A *= safe_exp(b_gJ[:, None] - b_gJ[None, :])
    b_A = tl.where(m_d & (m_J[:, None] & m_J[None, :]), b_A, 0.0) * b_bJ[:, None]
    b_Ai = -b_A
    for i in range(2, tl.minimum(16, p % 16 + 1)):
        b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A, 0.0), 0)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a = b_a + tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I

    sv = solved + (s * H + i_h) * SLD * SLD
    cu = cur + (i_n * H + i_h) * 16 * CHUNK
    r16 = tl.arange(0, 16)
    c64 = tl.arange(0, CHUNK)
    tl.store(
        cu + r16[:, None] * CHUNK + c64[None, :], tl.zeros([16, CHUNK], tl.float32)
    )
    tl.debug_barrier()
    _put(cu, 0, J, b_Ai, CHUNK)
    done = (p % 16 == 15) & (J <= 2)
    if done:
        _put(sv, J, J, b_Ai, SLD)

    def_gate = b_bJ[:, None]
    if J >= 1:
        b_g0 = tl.load(g_row + (0 * 16 + o_i) * H)
        b_A0 = _kk(k_rows, J, 0, p, Hg, K, BK) * safe_exp(b_gJ[:, None] - b_g0[None, :])
        b_A0 = b_A0 * def_gate
        b_Ai00 = _blk(sv, 0, 0, SLD)
        if J == 1:
            b_Ai10 = -_mm(_mm(b_Ai, b_A0), b_Ai00)
            _put(cu, 0, 0, b_Ai10, CHUNK)
            if done:
                _put(sv, 1, 0, b_Ai10, SLD)
        else:
            b_g1 = tl.load(g_row + (1 * 16 + o_i) * H)
            b_A1 = _kk(k_rows, J, 1, p, Hg, K, BK) * safe_exp(
                b_gJ[:, None] - b_g1[None, :]
            )
            b_A1 = b_A1 * def_gate
            b_Ai11 = _blk(sv, 1, 1, SLD)
            b_Ai10 = _blk(sv, 1, 0, SLD)
            if J == 2:
                b_Ai21 = -_mm(_mm(b_Ai, b_A1), b_Ai11)
                b_Ai20 = -_mm(b_Ai, _mm(b_A0, b_Ai00) + _mm(b_A1, b_Ai10))
                _put(cu, 0, 1, b_Ai21, CHUNK)
                _put(cu, 0, 0, b_Ai20, CHUNK)
                if done:
                    _put(sv, 2, 1, b_Ai21, SLD)
                    _put(sv, 2, 0, b_Ai20, SLD)
            else:
                b_g2 = tl.load(g_row + (2 * 16 + o_i) * H)
                b_A2 = _kk(k_rows, J, 2, p, Hg, K, BK) * safe_exp(
                    b_gJ[:, None] - b_g2[None, :]
                )
                b_A2 = b_A2 * def_gate
                b_Ai22 = _blk(sv, 2, 2, SLD)
                b_Ai21 = _blk(sv, 2, 1, SLD)
                b_Ai20 = _blk(sv, 2, 0, SLD)
                b_Ai32 = -_mm(_mm(b_Ai, b_A2), b_Ai22)
                b_Ai31 = -_mm(b_Ai, _mm(b_A1, b_Ai11) + _mm(b_A2, b_Ai21))
                b_Ai30 = -_mm(
                    b_Ai,
                    _mm(b_A0, b_Ai00) + _mm(b_A1, b_Ai10) + _mm(b_A2, b_Ai20),
                )
                _put(cu, 0, 2, b_Ai32, CHUNK)
                _put(cu, 0, 1, b_Ai31, CHUNK)
                _put(cu, 0, 0, b_Ai30, CHUNK)


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
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    """Rows p of w and u (recompute_w_u arithmetic, on a 16-row tile holding row p) into w_row[n, h], u_row[n, h]."""
    i_n, i_h = tl.program_id(0) // H, tl.program_id(0) % H
    s = tl.load(slots + i_n).to(tl.int64)
    p = tl.load(rows + i_n)
    k_rows = k_ring + (s * CHUNK * Hg + i_h // (H // Hg)) * K
    g_row = g_cum + i_n * CHUNK * H + i_h
    cu = cur + (i_n * H + i_h) * 16 * CHUNK
    rJ = p // 16 * 16 + tl.arange(0, 16)
    c64 = tl.arange(0, CHUNK)
    # recompute_w_u: u = A @ bf16(v * beta), w = A @ bf16(k * beta * exp(g)), with A in BF16 as the kkt kernel stores it.
    is_p = (rJ == p)[:, None]
    b_Ar = tl.where(is_p, tl.load(cu + (p % 16) * CHUNK + c64)[None, :], 0.0).to(
        tl.bfloat16
    )
    r64 = tl.arange(0, CHUNK)
    m_r = (r64 <= p)[:, None]
    b_beta = tl.load(beta_ring + (s * CHUNK + r64) * H + i_h, mask=r64 <= p, other=0.0)
    cv = tl.arange(0, V)
    b_v = tl.load(
        v_ring + (s * CHUNK * H + i_h) * V + r64[:, None] * (H * V) + cv[None, :],
        mask=m_r,
        other=0.0,
    )
    b_u = tl.dot(b_Ar, (b_v * b_beta[:, None]).to(b_v.dtype), allow_tf32=False)
    dst = u_row + (i_n * H + i_h) * V + cv[None, :] + rJ[:, None] * 0
    tl.store(dst, b_u.to(tl.bfloat16), mask=is_p)
    b_eg = tl.exp(tl.load(g_row + r64 * H))
    for i_k in tl.static_range(K // 64):
        ck = i_k * 64 + c64
        b_k = tl.load(
            k_rows + r64[:, None] * (Hg * K) + ck[None, :], mask=m_r, other=0.0
        )
        b_w = tl.dot(b_Ar, (b_k * b_beta[:, None] * b_eg[:, None]).to(b_k.dtype))
        dst = w_row + (i_n * H + i_h) * K + ck[None, :] + rJ[:, None] * 0
        tl.store(dst, b_w.to(tl.bfloat16), mask=is_p)


@triton.jit
def _out_kernel(
    state,
    k_ring,
    v_ring,
    beta_ring,
    vnew_ring,
    g_cum,
    solved,
    cur,
    w_row,
    u_row,
    q_in,
    out,
    slots,
    state_slots,
    rows,
    chunks,
    e_chunk,
    scale,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    ROWS: tl.constexpr,
    ONE_E: tl.constexpr,
    SLD: tl.constexpr,
    HAS_E: tl.constexpr,
):
    """Rows p of v_new (state kernel) and of the output (output kernel), on a ROWS-row tile holding row p (the other
    rows are zero); at row 63, the state kernel's step over the chunk (in place)."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    i_hg = i_h // (H // Hg)
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

    k_rows = k_ring + (s * CHUNK * Hg + i_hg) * K
    g_rows = g_cum + i_n * CHUNK * H + i_h
    b_G = tl.load(g_rows + r64 * H)
    b_beta = tl.load(beta_ring + (s * CHUNK + r64) * H + i_h, mask=r64 <= p, other=0.0)
    m_r = (r64 <= p)[:, None]
    cu = cur + (i_n * H + i_h) * 16 * CHUNK
    b_u = tl.where(is_p, tl.load(u_row + (i_n * H + i_h) * V + cv)[None, :], 0.0).to(
        tl.bfloat16
    )
    w_p = w_row + (i_n * H + i_h) * K
    ck = tl.arange(0, K)
    h0 = state + (ss * H + i_h) * V * K
    # State kernel: v_new = u - w @ bf16(E) (two K = 64 dots there; ONE_E does it as one K = 128 dot).
    if ONE_E:
        b_E = tl.load(h0 + cv[:, None] * K + ck[None, :])
        b_Eb = b_E.to(tl.bfloat16)
        b_w = tl.where(is_p, tl.load(w_p + ck)[None, :], 0.0).to(tl.bfloat16)
        b_vn = tl.dot(b_w, tl.trans(b_Eb))
    else:
        b_w1 = tl.where(is_p, tl.load(w_p + c64)[None, :], 0.0).to(tl.bfloat16)
        b_w2 = tl.where(is_p, tl.load(w_p + 64 + c64)[None, :], 0.0).to(tl.bfloat16)
        b_h1 = tl.load(h0 + cv[:, None] * K + c64[None, :])
        b_h2 = tl.load(h0 + cv[:, None] * K + 64 + c64[None, :])
        b_vn = tl.dot(b_w1, tl.trans(b_h1).to(tl.bfloat16))
        b_vn += tl.dot(b_w2, tl.trans(b_h2).to(tl.bfloat16))
    b_vn = b_u - b_vn
    vn_rows = vnew_ring + (s * CHUNK * H + i_h) * V
    tl.store(
        vn_rows + r_tile[:, None] * (H * V) + cv[None, :],
        b_vn.to(tl.bfloat16),
        mask=is_p,
    )
    tl.debug_barrier()

    # Output kernel (BT = 64, BK = 128): q @ bf16(E)^T and q @ K^T, gated, plus the causal intra-chunk term.
    b_q = tl.where(is_p, tl.load(q_in + (i_n * Hg + i_hg) * K + ck)[None, :], 0.0).to(
        tl.bfloat16
    )
    b_kT = tl.load(
        k_rows + r64[None, :] * (Hg * K) + ck[:, None],
        mask=(r64 <= p)[None, :],
        other=0.0,
    )
    if ONE_E:
        b_hb = b_Eb
    else:
        b_hb = tl.load(h0 + cv[:, None] * K + ck[None, :]).to(tl.bfloat16)
    b_o = tl.dot(b_q, tl.trans(b_hb))
    b_A = tl.dot(b_q, b_kT)
    b_gr = tl.load(g_rows + r_tile * H)
    b_o = b_o * exp(b_gr)[:, None]
    b_A = b_A * safe_exp(b_gr[:, None] - b_G[None, :])
    b_A = tl.where(r_tile[:, None] >= r64[None, :], b_A, 0)
    b_vt = tl.load(vn_rows + r64[:, None] * (H * V) + cv[None, :], mask=m_r, other=0.0)
    b_o = b_o * scale + tl.dot(b_A.to(b_vt.dtype), b_vt) * scale
    tl.store(
        out + (i_n * H + i_h) * V + cv[None, :] + r_tile[:, None] * 0,
        b_o.to(tl.bfloat16),
        mask=is_p,
    )

    if (p == CHUNK - 1) & (tl.load(state_slots + i_n) >= 0):
        # The chunk is complete: rows 0..47 of (I + A)^{-1} from the solved cache, rows 48..63 from cur.
        sv = solved + (s * H + i_h) * SLD * SLD
        lo = r64 < SOLVED_ROWS
        b_Af = tl.load(
            sv + r64[:, None] * SLD + c64[None, :],
            mask=lo[:, None] & (c64 < SOLVED_ROWS)[None, :],
            other=0.0,
        )
        b_Af += tl.load(
            cu + (r64[:, None] - SOLVED_ROWS) * CHUNK + c64[None, :],
            mask=(r64 >= SOLVED_ROWS)[:, None],
            other=0.0,
        )
        b_Af = b_Af.to(tl.bfloat16)
        b_v = tl.load(
            v_ring + (s * CHUNK * H + i_h) * V + r64[:, None] * (H * V) + cv[None, :]
        )
        b_eg = tl.exp(b_G)
        c_u = tl.dot(b_Af, (b_v * b_beta[:, None]).to(b_v.dtype), allow_tf32=False)
        c_g_last = tl.load(g_rows + (CHUNK - 1) * H)
        c_decay = exp(c_g_last)
        if ONE_E:
            b_k = tl.load(k_rows + r64[:, None] * (Hg * K) + ck[None, :])
            c_w = tl.dot(
                b_Af, (b_k * b_beta[:, None] * b_eg[:, None]).to(b_k.dtype)
            ).to(tl.bfloat16)
            c_v = tl.dot(c_w, tl.trans(b_Eb))
            c_v = c_u.to(tl.bfloat16) - c_v
            c_v = c_v * safe_exp(c_g_last - b_G)[:, None]
            # One FMA per element, as the state kernel compiles `h *= exp(g_last); h += (k^T v)^T` (see
            # chunk_rows2._close_head).
            c_h = tl.fma(
                b_E, c_decay, tl.trans(tl.dot(tl.trans(b_k), c_v.to(tl.bfloat16)))
            )
            tl.store(h0 + cv[:, None] * K + ck[None, :], c_h)
        else:
            b_k1 = tl.load(k_rows + r64[:, None] * (Hg * K) + c64[None, :])
            b_k2 = tl.load(k_rows + r64[:, None] * (Hg * K) + 64 + c64[None, :])
            c_w1 = tl.dot(b_Af, (b_k1 * b_beta[:, None] * b_eg[:, None]).to(b_k1.dtype))
            c_w2 = tl.dot(b_Af, (b_k2 * b_beta[:, None] * b_eg[:, None]).to(b_k2.dtype))
            c_w1 = c_w1.to(tl.bfloat16)
            c_w2 = c_w2.to(tl.bfloat16)
            c_v = tl.dot(c_w1, tl.trans(b_h1).to(tl.bfloat16))
            c_v += tl.dot(c_w2, tl.trans(b_h2).to(tl.bfloat16))
            c_v = c_u.to(tl.bfloat16) - c_v
            c_v = c_v * safe_exp(c_g_last - b_G)[:, None]
            c_vb = c_v.to(tl.bfloat16)
            c_h1 = tl.fma(b_h1, c_decay, tl.trans(tl.dot(tl.trans(b_k1), c_vb)))
            c_h2 = tl.fma(b_h2, c_decay, tl.trans(tl.dot(tl.trans(b_k2), c_vb)))
            tl.store(h0 + cv[:, None] * K + c64[None, :], c_h1)
            tl.store(h0 + cv[:, None] * K + 64 + c64[None, :], c_h2)
        if HAS_E and (i_v == 0):
            # E now stands at the next chunk (chunk_rows2's deferred-close record).
            tl.store(e_chunk + s * H + i_h, tl.load(chunks + i_n) + 1)


def kkt_autotuner():
    """The Autotuner under the kkt kernel's Heuristics wrapper."""
    fn = chunk_gated_delta_rule_fwd_kkt_solve_kernel
    while not isinstance(fn, triton.runtime.Autotuner):
        fn = fn.fn
    return fn


def kkt_config():
    """BK and num_warps of the config the prefill's autotuned kkt kernel runs with (the forward substitution's
    reductions follow num_warps). Raises before the prefill has run once, since the config is not known yet."""
    tuner = kkt_autotuner()
    if len(tuner.configs) == 1:
        best = tuner.configs[0]
    else:
        assert len(tuner.cache) == 1, (
            f"expected one autotuned kkt config, got {len(tuner.cache)}"
        )
        best = next(iter(tuner.cache.values()))
    return best.kwargs["BK"], best.num_warps


class Scratch:
    """Per-batch-row buffers of one step (shared by all layers)."""

    def __init__(self, max_bs, Hg, H, K, device):
        self.q = torch.empty(max_bs, Hg, K, dtype=torch.bfloat16, device=device)
        self.g_cum = torch.empty(max_bs, CHUNK, H, dtype=torch.float32, device=device)
        self.cur = torch.empty(max_bs, H, 16, CHUNK, dtype=torch.float32, device=device)
        self.w = torch.empty(max_bs, H, K, dtype=torch.bfloat16, device=device)
        self.u = torch.empty(max_bs, H, K, dtype=torch.bfloat16, device=device)


def step_slots(cache, indices):
    """(cache slots, state slots) of a step's batch rows, int32; computed once per step for all layers."""
    return cache.slots(indices).to(torch.int32), indices.to(torch.int32)


def decode(
    *,
    cache,
    scratch,
    state,
    slots,
    rows,
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    out,
    chunks=None,
    num_k_heads=16,
    head_k_dim=128,
    eps=1e-6,
    kkt=(64, 4),
    rows_tile=16,
    bv=64,
    one_e=True,
):
    """One token per request. mixed_qkv: [B, 2 * Hg * K + H * V] conv output (q, k before l2norm, then v); a, b:
    [B, H] gate inputs; rows: [B] int32 position of the token in its 64-token chunk; slots: step_slots(...);
    state: [slots, H, V, K] FP32 boundary states; out: [B, H, V].

    A cache with an ``e_chunk`` record (chunk_rows2.RowCache, whose solved cache is 64 x 64) also takes ``chunks``
    [B] int32, the chunk index of each token: the close at row 63 then records that E stands at the next chunk, so
    chunk_rows2 steps can follow on the same cache."""
    e_chunk = cache.e_chunk
    if e_chunk is not None and chunks is None:
        raise ValueError("a cache with e_chunk needs the chunk index of each row")
    sld = cache.solved.shape[-1]
    B, H = a.shape
    Hg, K = num_k_heads, head_k_dim
    V = (mixed_qkv.shape[1] - 2 * Hg * K) // H
    ring_slots, state_slots = slots
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
        eps,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
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
        H=H,
        Hg=Hg,
        K=K,
        BK=kkt[0],
        SLD=sld,
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
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        num_warps=4,
    )
    _out_kernel[(V // bv, B * H)](
        state,
        cache.k,
        cache.v,
        cache.beta,
        cache.v_new,
        scratch.g_cum,
        cache.solved,
        scratch.cur,
        scratch.w,
        scratch.u,
        scratch.q,
        out,
        ring_slots,
        state_slots,
        rows,
        rows if chunks is None else chunks,
        state_slots if e_chunk is None else e_chunk,
        K**-0.5,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BV=bv,
        ROWS=rows_tile,
        ONE_E=one_e,
        SLD=sld,
        HAS_E=e_chunk is not None,
        num_warps=4,
    )
    return out
