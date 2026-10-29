# SPDX-License-Identifier: Apache-2.0
"""One-launch GDN target-verify glue for the ReplaySSM circular-ring path.

Replaces, per GDN layer, the verify chain that runs between the in_proj GEMM
and the out_proj GEMM:

    fused_qkvzba_split_reshape_cat_contiguous   (pure copy of [q|k|v|z], [b|a])
    -> causal_conv1d_update                      (conv + SiLU, intermediate windows)
    -> gdn_replayssm_spec_circular_kernel        (L2-norm + ReplaySSM spec output)
    -> _layer_norm_fwd_1pass_kernel              (gated RMSNorm)

with one program per (request, value head) that reads the in_proj output rows
directly, recomputes the causal conv for its q/k/v channels, reconstructs the
spec-window output with the math of ``gdn_replayssm_spec_circular_kernel``
(same bf16 rounding points and ring entries) and applies the gated RMSNorm
before writing the out_proj input.

Latency structure (the chain is latency-bound, not bandwidth-bound):

* every global load is issued up front. The checkpoint tile, conv state, conv
  weights and replay cursors are written only by the previous step's commit,
  never by the in_proj GEMM that precedes this kernel, so they are loaded
  before ``griddepcontrol.wait``; the in_proj rows right after it. All stores
  come after all loads, so no load waits behind a may-alias store.
* the causal conv is computed once per channel from 7 vectors (3 state rows +
  T <= 4 token rows).
* the committed-history replay (write_pos > 0) is a uniform runtime branch; it
  is empty under fold-every-commit.

The per-request conv state is not rolled in place here: the post-verify commit
(``fused_conv_window_scatter_with_mask``) always rewrites the active conv state
from the intermediate window of the last accepted step, so only the window is
written. That keeps the q/k conv (shared by the value heads of a key head)
race-free across programs; q/k windows and ring keys come from the first value
head of each key group.

Linear-chain verify only (topk <= 1), like the circular ReplaySSM kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _conv4(
    xa,
    xb,
    xc,
    xd,
    w0,
    w1,
    w2,
    w3,
    bias,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # Same op order/rounding as _causal_conv1d_update_kernel / mk1 _conv_chunk.
    if HAS_BIAS:
        acc = bias
    else:
        acc = tl.zeros(w0.shape, dtype=tl.float32)
    acc += xa * w0
    acc += xb * w1
    acc += xc * w2
    acc += xd * w3
    if SILU:
        acc = acc / (1 + tl.exp(-acc))
    return acc.to(OUT_DTYPE).to(tl.float32)


@triton.jit
def _rows(o_s, r0, r1, r2, r3, T: tl.constexpr):
    # [BS, C] tile with row t = r_t for t < T, zero elsewhere.
    out = tl.where((o_s == 0)[:, None], r0[None, :], 0.0)
    if T > 1:
        out = tl.where((o_s == 1)[:, None], r1[None, :], out)
    if T > 2:
        out = tl.where((o_s == 2)[:, None], r2[None, :], out)
    if T > 3:
        out = tl.where((o_s == 3)[:, None], r3[None, :], out)
    return out


@triton.jit
def _store_win1(
    win_base,
    ch,
    t: tl.constexpr,
    a,
    b,
    c,
    stride_win_step,
    stride_win_dim,
    stride_win_w,
):
    p = win_base + t * stride_win_step + ch * stride_win_dim
    tl.store(p, a)
    tl.store(p + stride_win_w, b)
    tl.store(p + 2 * stride_win_w, c)


@triton.jit
def _store_windows(
    win_base,
    ch,
    s0,
    s1,
    s2,
    x0,
    x1,
    x2,
    x3,
    T: tl.constexpr,
    stride_win_step,
    stride_win_dim,
    stride_win_w,
):
    # Window after consuming token t = [xx[t+1], xx[t+2], xx[t+3]],
    # xx = [s0, s1, s2, x0, x1, x2, x3].
    _store_win1(
        win_base, ch, 0, s1, s2, x0, stride_win_step, stride_win_dim, stride_win_w
    )
    if T > 1:
        _store_win1(
            win_base, ch, 1, s2, x0, x1, stride_win_step, stride_win_dim, stride_win_w
        )
    if T > 2:
        _store_win1(
            win_base, ch, 2, x0, x1, x2, stride_win_step, stride_win_dim, stride_win_w
        )
    if T > 3:
        _store_win1(
            win_base, ch, 3, x1, x2, x3, stride_win_step, stride_win_dim, stride_win_w
        )


@triton.jit
def gdn_fused_verify_glue_kernel(
    qkvz,
    ba,
    conv_state,
    conv_w,
    conv_b,
    win,
    win_indices,
    A_log,
    dt_bias,
    norm_w,
    y,
    h0,
    d_cache,
    k_cache,
    g_cache,
    rawv_cache,
    rawk_cache,
    beta_cache,
    ssm_state_indices,
    replay_indices,
    write_pos,
    cache_base,
    is_flush_flags,
    scale,
    eps,
    stride_x_t,
    stride_ba_t,
    stride_y_t,
    stride_cs_slot,
    stride_cs_dim,
    stride_cs_tok,
    stride_w_dim,
    stride_w_width,
    stride_win_slot,
    stride_win_step,
    stride_win_dim,
    stride_win_w,
    stride_state_slot: tl.constexpr,
    stride_d_slot: tl.constexpr,
    stride_k_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BS: tl.constexpr,
    BC: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    STATE_FP32: tl.constexpr,
    STORE_EXACT_RAW: tl.constexpr,
    STORE_D_RESIDUAL: tl.constexpr,
    STORE_K_RESIDUAL: tl.constexpr,
    HAS_CONV_BIAS: tl.constexpr,
    CONV_SILU: tl.constexpr,
    NORM_ACT: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_GDC: tl.constexpr,
    PREFETCH: tl.constexpr,
    HIST_BRANCH: tl.constexpr,
    PDL_TRIGGER_EARLY: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_h = i_hv // (HV // H)
    ACT_DTYPE: tl.constexpr = qkvz.dtype.element_ty

    o_v = tl.arange(0, V)
    o_k = tl.arange(0, K)
    o_s = tl.arange(0, BS)
    o_c = tl.arange(0, BC)
    mask_s = o_s < T
    bos = (i_n * T).to(tl.int64)

    state_idx = tl.load(ssm_state_indices + i_n).to(tl.int64)
    replay_idx = tl.load(replay_indices + i_n).to(tl.int64)
    win_idx = tl.load(win_indices + i_n).to(tl.int64)
    is_null = state_idx <= NULL_BLOCK_ID
    state_safe = tl.where(is_null, 0, state_idx)
    replay_safe = tl.where(is_null | (replay_idx < 0), 0, replay_idx)

    if USE_GDC:
        if not PREFETCH:
            tl.extra.cuda.gdc_wait()
            if PDL_TRIGGER_EARLY:
                tl.extra.cuda.gdc_launch_dependents()

    # ---- loads not produced by the in_proj GEMM (previous-step state) ----
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    nw = tl.load(norm_w + o_v).to(tl.float32)
    b_is_flush = tl.load(is_flush_flags + replay_safe) != 0
    b_write_pos = tl.load(write_pos + replay_safe).to(tl.int64)
    b_cache_base = tl.load(cache_base + replay_safe).to(tl.int32)

    ch_q = i_h * K + o_k
    ch_k = H * K + i_h * K + o_k
    ch_v = 2 * H * K + i_hv * V + o_v
    pw_q = conv_w + ch_q * stride_w_dim
    pw_k = conv_w + ch_k * stride_w_dim
    pw_v = conv_w + ch_v * stride_w_dim
    wq0 = tl.load(pw_q)
    wq1 = tl.load(pw_q + stride_w_width)
    wq2 = tl.load(pw_q + 2 * stride_w_width)
    wq3 = tl.load(pw_q + 3 * stride_w_width)
    wk0 = tl.load(pw_k)
    wk1 = tl.load(pw_k + stride_w_width)
    wk2 = tl.load(pw_k + 2 * stride_w_width)
    wk3 = tl.load(pw_k + 3 * stride_w_width)
    wv0 = tl.load(pw_v)
    wv1 = tl.load(pw_v + stride_w_width)
    wv2 = tl.load(pw_v + 2 * stride_w_width)
    wv3 = tl.load(pw_v + 3 * stride_w_width)
    if HAS_CONV_BIAS:
        bq = tl.load(conv_b + ch_q).to(tl.float32)
        bk = tl.load(conv_b + ch_k).to(tl.float32)
        bv = tl.load(conv_b + ch_v).to(tl.float32)
    else:
        bq = 0.0
        bk = 0.0
        bv = 0.0
    s_slot = conv_state + state_safe * stride_cs_slot
    ps_q = s_slot + ch_q * stride_cs_dim
    ps_k = s_slot + ch_k * stride_cs_dim
    ps_v = s_slot + ch_v * stride_cs_dim
    sq0 = tl.load(ps_q)
    sq1 = tl.load(ps_q + stride_cs_tok)
    sq2 = tl.load(ps_q + 2 * stride_cs_tok)
    sk0 = tl.load(ps_k)
    sk1 = tl.load(ps_k + stride_cs_tok)
    sk2 = tl.load(ps_k + 2 * stride_cs_tok)
    sv0 = tl.load(ps_v)
    sv1 = tl.load(ps_v + stride_cs_tok)
    sv2 = tl.load(ps_v + 2 * stride_cs_tok)

    p_h0 = (
        h0
        + state_safe * stride_state_slot
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    sc_tile = tl.load(p_h0)

    if USE_GDC:
        if PREFETCH:
            tl.extra.cuda.gdc_wait()
            if PDL_TRIGGER_EARLY:
                tl.extra.cuda.gdc_launch_dependents()

    # ---- in_proj rows (produced by the predecessor) ----
    x_row = qkvz + bos * stride_x_t
    a_s = tl.load(
        ba + (bos + o_s) * stride_ba_t + HV + i_hv, mask=mask_s, other=0.0
    ).to(tl.float32)
    b_s = tl.load(ba + (bos + o_s) * stride_ba_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    xq0 = tl.load(x_row + ch_q)
    xk0 = tl.load(x_row + ch_k)
    xv0 = tl.load(x_row + ch_v)
    if T > 1:
        xq1 = tl.load(x_row + stride_x_t + ch_q)
        xk1 = tl.load(x_row + stride_x_t + ch_k)
        xv1 = tl.load(x_row + stride_x_t + ch_v)
    else:
        xq1 = xq0
        xk1 = xk0
        xv1 = xv0
    if T > 2:
        xq2 = tl.load(x_row + 2 * stride_x_t + ch_q)
        xk2 = tl.load(x_row + 2 * stride_x_t + ch_k)
        xv2 = tl.load(x_row + 2 * stride_x_t + ch_v)
    else:
        xq2 = xq0
        xk2 = xk0
        xv2 = xv0
    if T > 3:
        xq3 = tl.load(x_row + 3 * stride_x_t + ch_q)
        xk3 = tl.load(x_row + 3 * stride_x_t + ch_k)
        xv3 = tl.load(x_row + 3 * stride_x_t + ch_v)
    else:
        xq3 = xq0
        xk3 = xk0
        xv3 = xv0
    if NORM_ACT != 0:
        z = tl.load(
            x_row
            + o_s[None, :] * stride_x_t
            + 2 * H * K
            + HV * V
            + i_hv * V
            + o_v[:, None],
            mask=mask_s[None, :],
            other=0.0,
        ).to(tl.float32)  # [V, BS]

    p_y = y + (bos + o_s[None, :]) * stride_y_t + i_hv * V + o_v[:, None]  # [V, BS]
    if is_null:
        tl.store(
            p_y,
            tl.zeros([V, BS], dtype=tl.float32).to(p_y.dtype.element_ty),
            mask=mask_s[None, :],
        )
        return

    # ---- conv (+SiLU), once per channel ----
    q0 = _conv4(
        sq0, sq1, sq2, xq0, wq0, wq1, wq2, wq3, bq, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    q1 = _conv4(
        sq1, sq2, xq0, xq1, wq0, wq1, wq2, wq3, bq, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    q2 = _conv4(
        sq2, xq0, xq1, xq2, wq0, wq1, wq2, wq3, bq, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    q3 = _conv4(
        xq0, xq1, xq2, xq3, wq0, wq1, wq2, wq3, bq, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    k0 = _conv4(
        sk0, sk1, sk2, xk0, wk0, wk1, wk2, wk3, bk, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    k1 = _conv4(
        sk1, sk2, xk0, xk1, wk0, wk1, wk2, wk3, bk, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    k2 = _conv4(
        sk2, xk0, xk1, xk2, wk0, wk1, wk2, wk3, bk, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    k3 = _conv4(
        xk0, xk1, xk2, xk3, wk0, wk1, wk2, wk3, bk, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    v0 = _conv4(
        sv0, sv1, sv2, xv0, wv0, wv1, wv2, wv3, bv, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    v1 = _conv4(
        sv1, sv2, xv0, xv1, wv0, wv1, wv2, wv3, bv, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    v2 = _conv4(
        sv2, xv0, xv1, xv2, wv0, wv1, wv2, wv3, bv, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    v3 = _conv4(
        xv0, xv1, xv2, xv3, wv0, wv1, wv2, wv3, bv, HAS_CONV_BIAS, CONV_SILU, ACT_DTYPE
    )
    q_rows = _rows(o_s, q0, q1, q2, q3, T)  # [BS, K]
    k_rows = _rows(o_s, k0, k1, k2, k3, T)  # [BS, K]

    # ---- gates ----
    x = a_s + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_s = tl.where(mask_s, -tl.exp(A_log_val) * softplus_x, 0.0)
    beta_s = tl.where(mask_s, 1.0 / (1.0 + tl.exp(-b_s)), 0.0)
    G_s = tl.cumsum(g_s, axis=0)
    expG_s = tl.exp(G_s)

    # ---- FN1568 K64 norm and dot accumulation order ----
    # The FN circular kernel uses NK=2 at K128. A single K128 reduction/dot
    # changes compact-ring low parts, so retain the same K64 partial sums.
    BKT: tl.constexpr = 64
    NK: tl.constexpr = K // BKT
    qnorm_acc = tl.zeros([BS], dtype=tl.float32)
    knorm_acc = tl.zeros([BS], dtype=tl.float32)
    for kk in tl.static_range(NK):
        ids = tl.broadcast_to((kk * BKT + tl.arange(0, BKT))[None, :], (BS, BKT))
        qh = tl.gather(q_rows, ids, axis=1)
        kh = tl.gather(k_rows, ids, axis=1)
        qnorm_acc += tl.sum(qh * qh, axis=1)
        knorm_acc += tl.sum(kh * kh, axis=1)
    q_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(qnorm_acc + 1e-6), 0.0)
    k_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(knorm_acc + 1e-6), 0.0)
    q_precise = q_rows * (q_rnorm * scale)[:, None]
    k_precise = k_rows * k_rnorm[:, None]
    k_store_tile = k_precise.to(k_cache.dtype.element_ty)
    k_residual_tile = (k_precise - k_store_tile.to(tl.float32)).to(ACT_DTYPE)
    if STATE_FP32:
        q_tile = q_precise
        k_tile = k_precise
    else:
        q_tile = q_precise.to(ACT_DTYPE)
        k_tile = k_store_tile
    qT = tl.trans(q_tile)
    kT = tl.trans(k_tile)
    kk_mat = tl.zeros([BS, BS], dtype=tl.float32)
    kq_mat = tl.zeros([BS, BS], dtype=tl.float32)
    hw_q = tl.zeros([V, BS], dtype=tl.float32)
    hw_k = tl.zeros([V, BS], dtype=tl.float32)
    for kk in tl.static_range(NK):
        offs = kk * BKT + tl.arange(0, BKT)
        ids = tl.broadcast_to(offs[None, :], (BS, BKT))
        qh = tl.gather(q_tile, ids, axis=1)
        kh = tl.gather(k_tile, ids, axis=1)
        krh = tl.gather(k_residual_tile, ids, axis=1)
        sh = tl.gather(sc_tile, tl.broadcast_to(offs[None, :], (V, BKT)), axis=1)
        kk_mat += tl.dot(kh, tl.trans(kh), input_precision=DOT_PRECISION)
        kq_mat += tl.dot(kh, tl.trans(qh), input_precision=DOT_PRECISION)
        if STORE_K_RESIDUAL and not STATE_FP32:
            kk_mat += tl.dot(kh, tl.trans(krh), input_precision=DOT_PRECISION)
            kk_mat += tl.dot(krh, tl.trans(kh), input_precision=DOT_PRECISION)
            kk_mat += tl.dot(krh, tl.trans(krh), input_precision=DOT_PRECISION)
            kq_mat += tl.dot(krh, tl.trans(qh), input_precision=DOT_PRECISION)
        if STATE_FP32:
            shf = sh.to(tl.float32)
            hw_q += tl.dot(shf, tl.trans(qh), input_precision=DOT_PRECISION)
            hw_k += tl.dot(shf, tl.trans(kh), input_precision=DOT_PRECISION)
        else:
            shb = sh.to(kh.dtype)
            hw_q += tl.dot(shb, tl.trans(qh))
            hw_k += tl.dot(shb, tl.trans(kh))
            if STORE_K_RESIDUAL:
                hw_k += tl.dot(shb, tl.trans(krh))

    # ---- committed-history replay (empty under fold-every-commit) ----
    if HIST_BRANCH:
        has_hist = b_write_pos > 0
    else:
        has_hist = True
    if has_hist:
        cache_valid = o_c < b_write_pos
        phys_c = (b_cache_base + o_c) & (MAX_CACHE_LEN - 1)
        p_g_main = g_cache + replay_safe * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_c
        b_g_all = tl.load(p_g_main, mask=cache_valid, other=0.0).to(tl.float32)
        p_d_main = d_cache + (
            replay_safe * stride_d_slot
            + (i_hv * MAX_CACHE_LEN + phys_c[None, :]) * V
            + o_v[:, None]
        )
        b_d_all = tl.load(p_d_main, mask=cache_valid[None, :], other=0.0).to(tl.float32)
        p_k = k_cache + (
            replay_safe * stride_k_slot
            + (i_h * MAX_CACHE_LEN + phys_c[:, None]) * K
            + o_k[None, :]
        )
        khist_tile = tl.load(p_k, mask=cache_valid[:, None], other=0.0)
        b_g_prefix = tl.cumsum(b_g_all, axis=0)
        b_g_total = tl.sum(b_g_all, axis=0)
        b_replay_decay = tl.where(cache_valid, tl.exp(b_g_total - b_g_prefix), 0.0)
        b_total_decay = tl.exp(b_g_total)
        b_d_scaled_fp32 = b_d_all * b_replay_decay[None, :]
        if STATE_FP32:
            b_d_scaled = b_d_scaled_fp32
            khist_tile = khist_tile.to(tl.float32)
        else:
            b_d_scaled = b_d_scaled_fp32.to(d_cache.dtype.element_ty)
            khist_tile = khist_tile.to(k_cache.dtype.element_ty)
        scores_q = tl.dot(khist_tile, qT, input_precision=DOT_PRECISION)
        scores_k = tl.dot(khist_tile, kT, input_precision=DOT_PRECISION)
        hw_q = b_total_decay * hw_q + tl.dot(
            b_d_scaled, scores_q.to(b_d_scaled.dtype), input_precision=DOT_PRECISION
        )
        hw_k = b_total_decay * hw_k + tl.dot(
            b_d_scaled, scores_k.to(b_d_scaled.dtype), input_precision=DOT_PRECISION
        )

    # ---- strictly-lower A and T = (I + A)^{-1} ----
    lower = (o_s[:, None] > o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ij = G_s[:, None] - G_s[None, :]
    A_mat = tl.where(lower, beta_s[:, None] * tl.exp(diff_ij) * kk_mat, 0.0)
    b_Ai = -A_mat
    for ii in tl.static_range(2, T):
        row = tl.sum(tl.where((o_s == ii)[:, None], -A_mat, 0.0), axis=0)
        row = tl.where(o_s < ii, row, 0.0)
        row = row + tl.sum(row[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_s == ii)[:, None], row, b_Ai)
    T_mat = b_Ai + (o_s[:, None] == o_s[None, :]).to(tl.float32)

    # ---- R, D_spec = R T^T, DF = D F ----
    v_tile = tl.trans(_rows(o_s, v0, v1, v2, v3, T))  # [V, BS]
    R_mat = beta_s[None, :] * (v_tile - expG_s[None, :] * hw_k)
    causalF = (o_s[:, None] <= o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ji = G_s[None, :] - G_s[:, None]
    F_mat = tl.where(causalF, tl.exp(diff_ji) * kq_mat, 0.0)
    D_spec = tl.zeros([V, BS], dtype=tl.float32)
    for j in tl.static_range(T):
        Rj = tl.sum(tl.where((o_s == j)[None, :], R_mat, 0.0), axis=1)
        Tj = tl.sum(tl.where((o_s == j)[None, :], T_mat, 0.0), axis=1)
        D_spec += Rj[:, None] * Tj[None, :]
    DF = tl.zeros([V, BS], dtype=tl.float32)
    for j in tl.static_range(T):
        Dj = tl.sum(tl.where((o_s == j)[None, :], D_spec, 0.0), axis=1)
        Fj = tl.sum(tl.where((o_s == j)[:, None], F_mat, 0.0), axis=0)
        DF += Dj[:, None] * Fj[None, :]
    O_tile = expG_s[None, :] * hw_q + DF  # [V, BS]

    # ---- gated RMSNorm (norm before gate) from the activation-dtype output ----
    o_act = O_tile.to(ACT_DTYPE).to(tl.float32)
    xbar = tl.where(mask_s[None, :], o_act, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / V
    rstd = tl.rsqrt(var + eps)
    yv = o_act * rstd[None, :] * nw[:, None]
    if NORM_ACT == 1:
        yv *= z * tl.sigmoid(z)
    elif NORM_ACT == 2:
        yv *= tl.sigmoid(z)
    tl.store(p_y, yv.to(p_y.dtype.element_ty), mask=mask_s[None, :] & (~b_is_flush))
    if USE_GDC and not PDL_TRIGGER_EARLY:
        # The out_proj GEMM only reads y; let it launch once y is written.
        tl.extra.cuda.gdc_launch_dependents()

    # ---- windows (stock conv writes them for flush rows too) ----
    win_slot = win + win_idx * stride_win_slot
    write_k = i_hv == i_h * (HV // H)
    _store_windows(
        win_slot,
        ch_v,
        sv0,
        sv1,
        sv2,
        xv0,
        xv1,
        xv2,
        xv3,
        T,
        stride_win_step,
        stride_win_dim,
        stride_win_w,
    )
    if write_k:
        _store_windows(
            win_slot,
            ch_q,
            sq0,
            sq1,
            sq2,
            xq0,
            xq1,
            xq2,
            xq3,
            T,
            stride_win_step,
            stride_win_dim,
            stride_win_w,
        )
        _store_windows(
            win_slot,
            ch_k,
            sk0,
            sk1,
            sk2,
            xk0,
            xk1,
            xk2,
            xk3,
            T,
            stride_win_step,
            stride_win_dim,
            stride_win_w,
        )
    if b_is_flush:
        # verify-only spec launch leaves flush rows' outputs/rings untouched.
        return

    # ---- speculative ring entries at circular positions ----
    b_wp_i = b_write_pos.to(tl.int32)
    phys_spec = (b_cache_base + b_wp_i + o_s) & (MAX_CACHE_LEN - 1)
    spec_pos = b_write_pos + o_s
    spec_store_mask = mask_s & (spec_pos < MAX_CACHE_LEN)
    if write_k:
        spec_kt_mask = spec_store_mask[:, None]
        if STORE_EXACT_RAW or STORE_K_RESIDUAL:
            p_cur_rawk = rawk_cache + (
                replay_safe * stride_rawk_slot
                + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                + o_k[None, :]
            )
            if STORE_EXACT_RAW:
                tl.store(
                    p_cur_rawk,
                    k_rows.to(p_cur_rawk.dtype.element_ty),
                    mask=spec_kt_mask,
                )
            else:
                tl.store(p_cur_rawk, k_residual_tile, mask=spec_kt_mask)
        p_cur_k = k_cache + (
            replay_safe * stride_k_slot
            + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
            + o_k[None, :]
        )
        tl.store(p_cur_k, k_store_tile, mask=spec_kt_mask)
    p_cur_d = d_cache + (
        replay_safe * stride_d_slot
        + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
        + o_v[:, None]
    )
    d_hi = D_spec.to(p_cur_d.dtype.element_ty)
    tl.store(p_cur_d, d_hi, mask=spec_store_mask[None, :])
    if STORE_EXACT_RAW or STORE_D_RESIDUAL:
        p_cur_v = rawv_cache + (
            replay_safe * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
            + o_v[:, None]
        )
        if STORE_EXACT_RAW:
            tl.store(
                p_cur_v,
                v_tile.to(p_cur_v.dtype.element_ty),
                mask=spec_store_mask[None, :],
            )
        else:
            tl.store(
                p_cur_v,
                (D_spec - d_hi.to(tl.float32)).to(p_cur_v.dtype.element_ty),
                mask=spec_store_mask[None, :],
            )
    p_cur_g = g_cache + replay_safe * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_spec
    tl.store(p_cur_g, g_s, mask=spec_store_mask)
    if STORE_EXACT_RAW:
        p_cur_beta = (
            beta_cache
            + replay_safe * stride_beta_slot
            + i_hv * MAX_CACHE_LEN
            + phys_spec
        )
        tl.store(p_cur_beta, beta_s, mask=spec_store_mask)


_NORM_ACT = {None: 0, "": 0, "swish": 1, "silu": 1, "sigmoid": 2}


def gdn_fused_verify_glue(
    *,
    mixed_qkvz: torch.Tensor,  # [M, >= 2*H*K + 2*HV*V], unit last stride
    mixed_ba: torch.Tensor,  # [M, >= 2*HV], unit last stride
    conv_state: torch.Tensor,  # [slots, DIM, STATE_LEN]
    conv_weight: torch.Tensor,  # [DIM, 4]
    conv_bias: torch.Tensor | None,
    conv_activation,
    intermediate_conv_window: torch.Tensor,  # [islots, T, DIM, 3] (strided view)
    intermediate_state_indices: torch.Tensor,  # [>= B]
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,  # [V]
    norm_eps: float,
    norm_activation,
    checkpoint_state: torch.Tensor,  # [num_slots, HV, V, K]
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    rawv_cache: torch.Tensor | None,
    rawk_cache: torch.Tensor | None,
    beta_cache: torch.Tensor | None,
    cache_indices: torch.Tensor,  # [>= B]
    replay_indices: torch.Tensor,  # [>= B]
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    draft_token_num: int,
    out: torch.Tensor | None = None,
    null_block_id: int = -1,
    num_warps: int = 4,
    num_stages: int = 1,
    use_pdl: bool = True,
    pdl_trigger_early: bool = True,
    dot_precision: str = "tf32",
    prefetch: bool = True,
    hist_branch: bool = True,
    **_unused,
) -> torch.Tensor:
    """Returns the gated-RMSNorm output ``[M, HV*V]`` (the out_proj input)."""
    num_slots, HV, V, K = checkpoint_state.shape
    H = num_k_heads
    T = draft_token_num
    M = mixed_qkvz.shape[0]
    assert K == head_k_dim and M % T == 0
    B = M // T
    assert mixed_qkvz.stride(-1) == 1 and mixed_ba.stride(-1) == 1
    assert conv_weight.shape[-1] == 4, "fused verify glue supports conv width 4"
    state_len = conv_state.shape[-1]
    assert state_len == 3
    max_cache_len = d_cache.shape[-2]
    assert max_cache_len & (max_cache_len - 1) == 0
    assert 1 <= T <= 4, "fused verify glue supports 1..4 draft tokens"
    BS = max(4, triton.next_power_of_2(T))
    BC = max(16, triton.next_power_of_2(max_cache_len))
    if is_flush.dtype != torch.int8:
        is_flush = is_flush.to(torch.int8)
    if out is None:
        out = mixed_qkvz.new_empty(M, HV * V)
    if M == 0:
        return out

    store_exact_raw = (
        rawv_cache is not None and rawk_cache is not None and beta_cache is not None
    )
    store_residual = (
        rawv_cache is not None and rawk_cache is not None and beta_cache is None
    )
    rawv_buf = rawv_cache if rawv_cache is not None else d_cache
    rawk_buf = rawk_cache if rawk_cache is not None else k_cache
    beta_buf = beta_cache if beta_cache is not None else g_cache
    if isinstance(conv_activation, bool):
        conv_silu = conv_activation
    else:
        conv_silu = conv_activation in ("silu", "swish")
    w2 = conv_weight.reshape(conv_weight.shape[0], -1)
    extra = {"USE_GDC": True, "launch_pdl": True} if use_pdl else {"USE_GDC": False}
    grid = (B, HV)
    gdn_fused_verify_glue_kernel[grid](
        mixed_qkvz,
        mixed_ba,
        conv_state,
        w2,
        conv_bias if conv_bias is not None else w2,
        intermediate_conv_window,
        intermediate_state_indices,
        A_log,
        dt_bias,
        norm_weight,
        out,
        checkpoint_state,
        d_cache,
        k_cache,
        g_cache,
        rawv_buf,
        rawk_buf,
        beta_buf,
        cache_indices,
        replay_indices,
        write_pos,
        cache_base,
        is_flush,
        K**-0.5,
        norm_eps,
        mixed_qkvz.stride(0),
        mixed_ba.stride(0),
        out.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        w2.stride(0),
        w2.stride(1),
        intermediate_conv_window.stride(0),
        intermediate_conv_window.stride(1),
        intermediate_conv_window.stride(2),
        intermediate_conv_window.stride(3),
        checkpoint_state.stride(0),
        d_cache.stride(0),
        k_cache.stride(0),
        g_cache.stride(0),
        rawv_buf.stride(0),
        rawk_buf.stride(0),
        beta_buf.stride(0),
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BS=BS,
        BC=BC,
        MAX_CACHE_LEN=max_cache_len,
        STATE_LEN=state_len,
        SOFTPLUS_THRESHOLD=20.0,
        STATE_FP32=checkpoint_state.dtype == torch.float32,
        STORE_EXACT_RAW=store_exact_raw,
        STORE_D_RESIDUAL=store_residual,
        STORE_K_RESIDUAL=store_residual,
        HAS_CONV_BIAS=conv_bias is not None,
        CONV_SILU=conv_silu,
        NORM_ACT=_NORM_ACT[norm_activation],
        NULL_BLOCK_ID=null_block_id,
        DOT_PRECISION=dot_precision,
        PREFETCH=prefetch,
        HIST_BRANCH=hist_branch,
        PDL_TRIGGER_EARLY=pdl_trigger_early,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra,
    )
    return out
