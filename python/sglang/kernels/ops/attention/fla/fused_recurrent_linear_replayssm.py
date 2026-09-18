# Buffered output-only linear-attention decode (ReplaySSM Part A), ported to
# SGLang. Covers BOTH gate granularities with one kernel:
#   * GDN  (``IS_KDA=False``): per-head SCALAR gate ``alpha = exp(g)``.
#   * KDA  (``IS_KDA=True``):  per-K-channel gate ``alpha[k] = exp(g[k])`` —
#     the state decays column-wise, ``S' = S . Diag(alpha) + d k^T``.
# GDN is the special case of KDA with all per-K decays equal; a single
# ``IS_KDA`` constexpr selects the gate path and the GDN path is bit-for-bit
# the original (no regression to the validated GDN kernel).
#
# This is a STANDALONE increment: kernel + wrapper only. It is NOT yet wired
# into the memory pool / radix cache / scheduler / backend dispatch. The caller
# (currently the correctness test and the microbenchmark) owns the ring tensors.
#
# Idea (vs. ``fused_recurrent_gated_delta_rule_packed_decode`` / ``..._kda_...``):
#   The plain packed decode reads the full recurrent state S [HV, V, K] from
#   HBM and writes it back *every* decode step (~8*d*n bytes/step of state
#   traffic for an fp32 state, read+write).  ReplaySSM keeps a small per-slot
#   ring buffer of the last L steps' (d, k, g) and only WRITES the full state
#   every L steps (a "flush"); on non-flush steps it appends a tiny
#   (d, k, g) record and reconstructs the readout from the checkpoint S0 plus
#   the buffer.  S0 is still READ every step, so per-step state traffic drops
#   from read+write (~8*d*n) to read-only (~4*d*n) -> roughly halved.
#
# Math (single head, single step; matches the packed decode kernels exactly).
# Let ``a = exp(g)`` be the decay (scalar for GDN, per-K vector for KDA) and
# ``S`` the state *before* this token:
#   d_cur = beta * (v - (S . Diag(a)) . k) = beta * (v - S . (a (.) k))
#   o     = (S . Diag(a)) . q + d_cur*(k^T q) = S . (a (.) q) + d_cur*(k^T q)
#   S_new = S . Diag(a) + d_cur k^T          # only persisted on flush
# where ``(.)`` is elementwise over K.  For GDN ``a`` is scalar so
# ``S . (a (.) q) = a * (S . q)`` (the cheap scalar post-multiply); for KDA the
# per-K ``a`` folds into q/k before the matvec.  k^T q uses the RAW current
# k/q (the rank-1 term), so it is identical for both gate types.
#
# Buffered reconstruction: with buffered steps j=0..m-1 holding (d_j, k_j, g_j),
# the state *before* the current token is
#   S = Diag(A) . S0 + sum_j d_j (W_j (.) k_j)^T          (per-K form)
# with A[c]   = exp(sum_j g_j[c])                  (total decay, per-K)
#      W_j[c] = exp(sum_i g_i[c] - cumsum_inclusive_j[c]) = prod_{i>j} a_i[c].
# For GDN A and W_j are scalars (g is K-independent) and W_j folds onto d_j
# instead of k_j (either factor works for a scalar). S is reconstructed in
# K-tiles and immediately read with q (and k) -> the [V,K] state tile is never
# fully materialized to HBM on a non-flush step.
#
# At L=1 the ring is always empty and ``write_pos == L-1`` every step, so the
# reconstruction term is zero, the total decay is 1, and this kernel reduces
# *algebraically* to the corresponding packed-decode kernel.
#
# SPDX-License-Identifier: Apache-2.0
# Ported from vllm/model_executor/layers/fla/ops/fused_recurrent_replayssm.py
# (ReplaySSM, commit 3c85112) and adapted to SGLang's packed GDN/KDA decode
# layout.

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_recurrent_linear_replayssm_decode_kernel(
    mixed_qkv,  # [B, 2*H*K + HV*V] packed (q | k | v) after conv1d
    a,  # GDN: [B, HV] gate input ; KDA: [B, HV, K] per-K gate input
    b,  # [B, HV] beta input b
    A_log,  # [HV] log-space decay parameter (per-head scalar, both gate types)
    dt_bias,  # GDN: [HV] ; KDA: [HV, K] time-step bias
    o,  # [B, HV, V] output (written every step)
    h0,  # [num_slots, HV, V, K] checkpoint state (read every step)
    ht,  # [num_slots, HV, V, K] checkpoint state (written only on flush; == h0)
    d_cache,  # [num_slots, HV, L, V] ring: corrected delta vectors
    k_cache,  # [num_slots, H,  L, K] ring: (normed/scaled) keys
    g_cache,  # GDN: [num_slots, HV, L] ; KDA: [num_slots, HV, L, K] log-decay gates (fp32)
    ssm_state_indices,  # [B] physical state slot per decode row
    write_pos,  # [B] int32 per-row ring cursor (0..L-1)
    force_flush,  # [B] int32: !=0 forces a flush this step (radix track boundary)
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr,
    BKT: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    HAS_FORCE_FLUSH: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)

    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, BC)
    mask_v = o_v < V

    # Resolve the physical state slot; zero the output and bail for padded rows.
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    if state_idx < 0:
        tl.store(
            p_o,
            tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty),
            mask=mask_v,
        )
        return

    # Per-row buffer cursor and flush flag (device-side branch, no host branch),
    # plus the set of valid (already committed) cache positions.
    b_write_pos = tl.load(write_pos + i_n).to(tl.int64)
    b_is_flush = b_write_pos == MAX_CACHE_LEN - 1
    if HAS_FORCE_FLUSH:
        # A radix track-boundary (or any caller-forced) flush folds the partial
        # ring (the real `write_pos` entries, NOT L-1) + current token into the
        # checkpoint so an external snapshot reads an up-to-date state. cache_valid
        # below still uses the true write_pos, so only committed entries are read.
        b_is_flush = b_is_flush | (tl.load(force_flush + i_n) != 0)
    cache_valid = o_c < b_write_pos

    # Gate for the current token.  beta is a per-head scalar for both gate
    # types; A_log is a per-head scalar for both.  The decay g/alpha is a
    # per-head scalar for GDN (computed here) and a per-K vector for KDA
    # (computed per K-tile inside the loop, since it is K-indexed).
    #   g = -exp(A_log) * softplus(a + dt_bias);  alpha = exp(g);  beta = sigmoid(b)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)
    if not IS_KDA:
        a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g_val = -tl.exp(A_log_val) * softplus_x
        alpha_val = tl.exp(g_val)

        # Replay decay over the committed cache, from the cached per-step gates.
        # b_replay_decay[j] = exp(sum_i g_i - cumsum_inclusive_j) = prod_{i>j} alpha_i
        p_g_main = g_cache + (state_idx * HV + i_hv) * MAX_CACHE_LEN + o_c
        b_g_all = tl.load(p_g_main, mask=cache_valid, other=0.0).to(tl.float32)
        b_g_prefix = tl.cumsum(b_g_all, axis=0)
        b_g_total = tl.sum(b_g_all, axis=0)
        b_replay_decay = tl.where(cache_valid, tl.exp(b_g_total - b_g_prefix), 0.0)
        b_total_decay = tl.exp(b_g_total)

    # Cached corrected-delta vectors d (K-independent).  Layout
    # d_cache[slot, hv, L, V] -> index [V, BC] tile.
    p_d_main = d_cache + (
        ((state_idx * HV + i_hv) * MAX_CACHE_LEN + o_c[None, :]) * V + o_v[:, None]
    )
    b_d_all = tl.load(
        p_d_main, mask=mask_v[:, None] & cache_valid[None, :], other=0
    ).to(tl.float32)
    # Cast the (d, k) reconstruction-dot operands to the I/O dtype so tl.dot
    # runs on TENSOR CORES (bf16 tensor cores for bf16, TF32 for fp32). This is
    # the ReplaySSM reference path and is performance-critical: the L-deep
    # reconstruction is L x the baseline's rank-1 compute, which only stays
    # hidden under the (reduced) memory traffic if it runs on tensor cores. An
    # IEEE fp32 dot here disables tensor cores and makes the kernel ~10-20x
    # slower. Tensor-core precision (~4e-4 TF32 / ~1e-3 bf16) is benign
    # end-to-end (ReplaySSM bf16 GSM8K parity); the unit test uses
    # tensor-core-realistic tolerances. At L=1 the buffer is empty so this dot
    # is identically zero and the path stays bit-exact regardless of precision.
    # GDN folds the (scalar) replay decay onto d here; KDA folds the (per-K)
    # replay decay onto the cached keys inside the K-tile loop instead.
    if not IS_KDA:
        b_d_tc = (b_d_all * b_replay_decay[None, :]).to(
            p_o.dtype.element_ty
        )  # [BV, BC]
    else:
        b_d_tc = b_d_all.to(p_o.dtype.element_ty)  # [BV, BC]

    # Current token value (for the delta-rule update).
    v_off = (2 * H * K) + i_hv * V + o_v
    b_v = tl.load(
        mixed_qkv + i_n * stride_mixed_qkv_tok + v_off, mask=mask_v, other=0
    ).to(tl.float32)

    # Optional q/k L2 norm: full-vector reciprocal norms (computed, not kept).
    if USE_QK_L2NORM_IN_KERNEL:
        o_kf = tl.arange(0, BK)
        mask_kf = o_kf < K
        p_mix = mixed_qkv + i_n * stride_mixed_qkv_tok
        qf = tl.load(p_mix + i_h * K + o_kf, mask=mask_kf, other=0).to(tl.float32)
        kf = tl.load(p_mix + H * K + i_h * K + o_kf, mask=mask_kf, other=0).to(
            tl.float32
        )
        q_rnorm = 1.0 / tl.sqrt(tl.sum(qf * qf) + 1e-6)
        k_rnorm = 1.0 / tl.sqrt(tl.sum(kf * kf) + 1e-6)
    else:
        q_rnorm = 1.0
        k_rnorm = 1.0

    # Reconstruct S from the checkpoint + cached (d, k) in K-tiles and read it
    # with the current (scaled) q and k.  K-tiling keeps the per-program tile
    # small so the full [V, K] state is never materialized.  Also append the
    # current key chunk to the ring cache (non-flush only).
    b_state_q = tl.zeros([BV], dtype=tl.float32)
    b_state_k = tl.zeros([BV], dtype=tl.float32)
    cur_kq = tl.zeros([1], dtype=tl.float32)
    write_k = (not b_is_flush) and (i_v == 0) and (i_hv == i_h * (HV // H))
    write_g_kda = IS_KDA and (not b_is_flush) and (i_v == 0)
    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        p_mix = mixed_qkv + i_n * stride_mixed_qkv_tok
        q_c = (
            tl.load(p_mix + i_h * K + o_kt, mask=mask_kt, other=0).to(tl.float32)
            * q_rnorm
        )
        k_c = (
            tl.load(p_mix + H * K + i_h * K + o_kt, mask=mask_kt, other=0).to(
                tl.float32
            )
            * k_rnorm
        )
        q_cs = q_c * scale
        # Rank-1 output term uses the RAW current k/q (gate-independent).
        cur_kq += tl.sum(k_c * q_cs)

        # This K-tile of the state: S_tile = Diag(A_tile) S0_tile + d (.) (W (.) k_cache).
        p_h0_c = (
            h0
            + state_idx * stride_init_state_token
            + i_hv * V * K
            + o_v[:, None] * K
            + o_kt[None, :]
        )
        b_h0_c = tl.load(p_h0_c, mask=mask_v[:, None] & mask_kt[None, :], other=0).to(
            tl.float32
        )
        p_k_c = (
            k_cache
            + ((state_idx * H + i_h) * MAX_CACHE_LEN + o_c[:, None]) * K
            + o_kt[None, :]
        )

        if not IS_KDA:
            b_k_all_c = tl.load(
                p_k_c, mask=cache_valid[:, None] & mask_kt[None, :], other=0
            ).to(p_o.dtype.element_ty)
            b_h_c = b_h0_c * b_total_decay + tl.dot(b_d_tc, b_k_all_c).to(tl.float32)
            # GDN: scalar current-token decay applied after the loop.
            q_eff = q_cs
            k_eff = k_c
        else:
            # KDA per-K decay: load this tile's cached gates [BC, BKT], form the
            # per-K total / replay decay, fold the replay decay onto the cached
            # keys and the total decay onto S0.
            p_g_c = (
                g_cache
                + ((state_idx * HV + i_hv) * MAX_CACHE_LEN + o_c[:, None]) * K
                + o_kt[None, :]
            )
            b_g_all_c = tl.load(
                p_g_c, mask=cache_valid[:, None] & mask_kt[None, :], other=0.0
            ).to(tl.float32)
            b_g_prefix_c = tl.cumsum(b_g_all_c, axis=0)  # [BC, BKT]
            b_g_total_c = tl.sum(b_g_all_c, axis=0)  # [BKT]
            b_replay_decay_c = tl.where(
                cache_valid[:, None],
                tl.exp(b_g_total_c[None, :] - b_g_prefix_c),
                0.0,
            )  # [BC, BKT]
            b_total_decay_c = tl.exp(b_g_total_c)  # [BKT]
            b_k_all_c = tl.load(
                p_k_c, mask=cache_valid[:, None] & mask_kt[None, :], other=0.0
            ).to(tl.float32)
            b_k_scaled = (b_k_all_c * b_replay_decay_c).to(p_o.dtype.element_ty)
            b_h_c = b_h0_c * b_total_decay_c[None, :] + tl.dot(b_d_tc, b_k_scaled).to(
                tl.float32
            )
            # KDA: current-token per-K decay folds into q/k for the readout.
            p_a_c = a + i_n * stride_a_tok + i_hv * K + o_kt
            p_dt_c = dt_bias + i_hv * K + o_kt
            b_a_c = tl.load(p_a_c, mask=mask_kt, other=0.0).to(tl.float32)
            b_dt_c = tl.load(p_dt_c, mask=mask_kt, other=0.0).to(tl.float32)
            x_c = b_a_c + b_dt_c
            softplus_c = tl.where(
                x_c <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x_c)), x_c
            )
            g_cur_c = -tl.exp(A_log_val) * softplus_c  # [BKT]
            alpha_cur_c = tl.exp(g_cur_c)
            q_eff = q_cs * alpha_cur_c
            k_eff = k_c * alpha_cur_c

            # Append this tile of the current gate to the ring (non-flush only).
            if write_g_kda:
                p_cur_g = (
                    g_cache
                    + ((state_idx * HV + i_hv) * MAX_CACHE_LEN + b_write_pos) * K
                    + o_kt
                )
                tl.store(p_cur_g, g_cur_c, mask=mask_kt & (b_write_pos < MAX_CACHE_LEN))

        # Read the state with the (gate-folded) q and k, accumulated across tiles.
        b_state_q += tl.sum(b_h_c * q_eff[None, :], axis=1)
        b_state_k += tl.sum(b_h_c * k_eff[None, :], axis=1)

        if write_k:
            p_cur_k = (
                k_cache
                + ((state_idx * H + i_h) * MAX_CACHE_LEN + b_write_pos) * K
                + o_kt
            )
            tl.store(
                p_cur_k,
                k_c.to(p_o.dtype.element_ty),
                mask=mask_kt & (b_write_pos < MAX_CACHE_LEN),
            )

    # Current-token output: (S . Diag(a)) q + d_cur*(k . q), with the new
    # corrected delta-rule vector d_cur = beta * (v - (S . Diag(a)) k).
    # For GDN the per-head scalar decay is applied here; for KDA it was already
    # folded into q/k above.
    if not IS_KDA:
        b_state_q *= alpha_val
        b_state_k *= alpha_val
    b_d_cur = beta_val * (b_v - b_state_k)
    b_o = b_state_q + b_d_cur * tl.sum(cur_kq)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    if b_is_flush:
        # Flush: fold the current token into the checkpoint, S_new = S.Diag(a) +
        # d_cur k^T, and persist it.  Re-walk K chunks to rebuild S, then apply
        # the update.  After this the ring is logically cleared (the caller
        # resets write_pos to 0 on the next step).
        for kk in range(NK):
            o_kt = kk * BKT + tl.arange(0, BKT)
            mask_kt = o_kt < K
            p_mix = mixed_qkv + i_n * stride_mixed_qkv_tok
            k_c = (
                tl.load(p_mix + H * K + i_h * K + o_kt, mask=mask_kt, other=0).to(
                    tl.float32
                )
                * k_rnorm
            )
            p_h0_c = (
                h0
                + state_idx * stride_init_state_token
                + i_hv * V * K
                + o_v[:, None] * K
                + o_kt[None, :]
            )
            b_h0_c = tl.load(
                p_h0_c, mask=mask_v[:, None] & mask_kt[None, :], other=0
            ).to(tl.float32)
            p_k_c = (
                k_cache
                + ((state_idx * H + i_h) * MAX_CACHE_LEN + o_c[:, None]) * K
                + o_kt[None, :]
            )
            if not IS_KDA:
                b_k_all_c = tl.load(
                    p_k_c, mask=cache_valid[:, None] & mask_kt[None, :], other=0
                ).to(p_o.dtype.element_ty)
                b_h_c = b_h0_c * b_total_decay + tl.dot(b_d_tc, b_k_all_c).to(
                    tl.float32
                )
                b_h_new_c = alpha_val * b_h_c + b_d_cur[:, None] * k_c[None, :]
            else:
                p_g_c = (
                    g_cache
                    + ((state_idx * HV + i_hv) * MAX_CACHE_LEN + o_c[:, None]) * K
                    + o_kt[None, :]
                )
                b_g_all_c = tl.load(
                    p_g_c, mask=cache_valid[:, None] & mask_kt[None, :], other=0.0
                ).to(tl.float32)
                b_g_prefix_c = tl.cumsum(b_g_all_c, axis=0)
                b_g_total_c = tl.sum(b_g_all_c, axis=0)
                b_replay_decay_c = tl.where(
                    cache_valid[:, None],
                    tl.exp(b_g_total_c[None, :] - b_g_prefix_c),
                    0.0,
                )
                b_total_decay_c = tl.exp(b_g_total_c)
                b_k_all_c = tl.load(
                    p_k_c, mask=cache_valid[:, None] & mask_kt[None, :], other=0.0
                ).to(tl.float32)
                b_k_scaled = (b_k_all_c * b_replay_decay_c).to(p_o.dtype.element_ty)
                b_h_c = b_h0_c * b_total_decay_c[None, :] + tl.dot(
                    b_d_tc, b_k_scaled
                ).to(tl.float32)
                p_a_c = a + i_n * stride_a_tok + i_hv * K + o_kt
                p_dt_c = dt_bias + i_hv * K + o_kt
                b_a_c = tl.load(p_a_c, mask=mask_kt, other=0.0).to(tl.float32)
                b_dt_c = tl.load(p_dt_c, mask=mask_kt, other=0.0).to(tl.float32)
                x_c = b_a_c + b_dt_c
                softplus_c = tl.where(
                    x_c <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x_c)), x_c
                )
                alpha_cur_c = tl.exp(-tl.exp(A_log_val) * softplus_c)  # [BKT]
                b_h_new_c = (
                    b_h_c * alpha_cur_c[None, :] + b_d_cur[:, None] * k_c[None, :]
                )
            p_ht_c = (
                ht
                + state_idx * stride_final_state_token
                + i_hv * V * K
                + o_v[:, None] * K
                + o_kt[None, :]
            )
            tl.store(
                p_ht_c,
                b_h_new_c.to(p_ht_c.dtype.element_ty),
                mask=mask_v[:, None] & mask_kt[None, :],
            )
    else:
        # Non-flush: append the current token's corrected delta d to the cache
        # (k chunks were written inside the loop; KDA's g chunks too).  GDN's
        # scalar g is appended here.
        p_cur_d = (
            d_cache + ((state_idx * HV + i_hv) * MAX_CACHE_LEN + b_write_pos) * V + o_v
        )
        tl.store(
            p_cur_d,
            b_d_cur.to(p_cur_d.dtype.element_ty),
            mask=mask_v & (b_write_pos < MAX_CACHE_LEN),
        )
        if (not IS_KDA) and (i_v == 0):
            p_cur_g = g_cache + (state_idx * HV + i_hv) * MAX_CACHE_LEN + b_write_pos
            tl.store(p_cur_g, g_val, mask=b_write_pos < MAX_CACHE_LEN)


def fused_recurrent_linear_replayssm_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    write_pos: torch.Tensor,
    force_flush: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    block_v: int | None = None,
    num_warps: int = 1,
    num_stages: int = 3,
    nk: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Buffered output-only linear-attention autoregressive decode (1 token/seq).

    One kernel for both gate granularities, selected by ``is_kda``:
      * ``is_kda=False`` (GDN): per-head SCALAR gate.  ``a``=[B, HV],
        ``dt_bias``=[HV], ``g_cache``=[num_slots, HV, L].
      * ``is_kda=True`` (KDA): per-K-channel gate.  ``a``=[B, HV, K],
        ``dt_bias``=[HV, K], ``g_cache``=[num_slots, HV, L, K].
    ``A_log`` is [HV] (per-head scalar) for both.

    Same call surface as the packed decode plus the three ring caches
    (``d_cache`` / ``k_cache`` / ``g_cache``) and the per-decode-row
    ``write_pos`` cursor.  ``initial_state`` is both the checkpoint read (h0)
    and the (flush-only) checkpoint write (ht), in place.

    Allocates nothing persistent: the caller owns the ring tensors and is
    responsible for advancing / resetting ``write_pos`` (e.g. ``(write_pos+1) %
    L`` after each step).  This is a STANDALONE kernel; the memory-pool / cache
    integration is a later phase.
    """
    if mixed_qkv.ndim != 2:
        raise ValueError(f"`mixed_qkv` must be 2D (got ndim={mixed_qkv.ndim}).")
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if b.ndim != 2:
        raise ValueError(f"`b` must be 2D (got b.ndim={b.ndim}).")
    if A_log.ndim != 1:
        raise ValueError("`A_log` must be a 1D tensor.")
    if initial_state.ndim != 4:
        raise ValueError(f"`initial_state` must be 4D (got ndim={initial_state.ndim}).")
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")
    if write_pos.ndim != 1 or write_pos.dtype != torch.int32:
        raise ValueError("`write_pos` must be a 1D int32 tensor.")
    if force_flush is not None and (
        force_flush.ndim != 1 or force_flush.dtype != torch.int32
    ):
        raise ValueError("`force_flush` must be a 1D int32 tensor or None.")

    B = mixed_qkv.shape[0]
    num_state_slots, HV, V, K = initial_state.shape
    qkv_dim = mixed_qkv.shape[1]
    q_dim = (qkv_dim - HV * V) // 2
    if q_dim <= 0 or q_dim % K != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}, K={K}."
        )
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )
    max_cache_len = d_cache.shape[2]

    # Gate-shape sanity: GDN scalar gate vs KDA per-K gate.
    if is_kda:
        if a.ndim != 3 or tuple(a.shape) != (B, HV, K):
            raise ValueError(
                f"KDA `a` must have shape {(B, HV, K)} (got {tuple(a.shape)})."
            )
        if dt_bias.ndim != 2 or tuple(dt_bias.shape) != (HV, K):
            raise ValueError(
                f"KDA `dt_bias` must have shape {(HV, K)} (got {tuple(dt_bias.shape)})."
            )
        if not a.is_contiguous() or not dt_bias.is_contiguous():
            raise ValueError("KDA `a`/`dt_bias` must be contiguous.")
        g_expect = (HV, max_cache_len, K)
    else:
        if a.ndim != 2 or tuple(a.shape) != (B, HV):
            raise ValueError(
                f"GDN `a` must have shape {(B, HV)} (got {tuple(a.shape)})."
            )
        if dt_bias.ndim != 1 or dt_bias.shape[0] != HV:
            raise ValueError(
                f"GDN `dt_bias` must have shape {(HV,)} (got {tuple(dt_bias.shape)})."
            )
        g_expect = (HV, max_cache_len)

    # Cache shape sanity (per state slot): d=(HV, L, V), k=(H, L, K).
    if tuple(d_cache.shape[1:]) != (HV, max_cache_len, V):
        raise ValueError(
            f"`d_cache` per-slot shape must be {(HV, max_cache_len, V)} "
            f"(got {tuple(d_cache.shape[1:])})."
        )
    if tuple(k_cache.shape[1:]) != (H, max_cache_len, K):
        raise ValueError(
            f"`k_cache` per-slot shape must be {(H, max_cache_len, K)} "
            f"(got {tuple(k_cache.shape[1:])})."
        )
    if tuple(g_cache.shape[1:]) != g_expect:
        raise ValueError(
            f"`g_cache` per-slot shape must be {g_expect} "
            f"(got {tuple(g_cache.shape[1:])})."
        )
    if g_cache.dtype != torch.float32:
        raise ValueError(f"`g_cache` must be float32 (got {g_cache.dtype}).")
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got {tuple(out.shape)})."
        )
    if write_pos.shape[0] != B or ssm_state_indices.shape[0] != B:
        raise ValueError(
            "`write_pos` and `ssm_state_indices` must both have length B="
            f"{B} (got {write_pos.shape[0]}, {ssm_state_indices.shape[0]})."
        )

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(
            f"Cached decode kernel only supports NK_global=1 (got K={K}, BK={BK})."
        )
    if BK % nk != 0:
        raise ValueError(f"nk={nk} must divide BK={BK}.")
    BKT = BK // nk
    if BKT < 16:
        raise ValueError(f"BKT={BKT} must be >=16 for tl.dot (nk={nk}, BK={BK}).")
    # K-tiling keeps the per-program tile small enough that a larger BV fits
    # without register spilling -> NV=1 -> half the grid -> fewer redundant
    # cache / metadata loads.
    BV = block_v if block_v is not None else min(triton.next_power_of_2(V), 64)
    BC = max(16, triton.next_power_of_2(max_cache_len))

    grid = (triton.cdiv(V, BV), B, HV)
    fused_recurrent_linear_replayssm_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        d_cache=d_cache,
        k_cache=k_cache,
        g_cache=g_cache,
        ssm_state_indices=ssm_state_indices,
        write_pos=write_pos,
        force_flush=force_flush if force_flush is not None else write_pos,
        scale=scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BC=BC,
        NK=nk,
        BKT=BKT,
        MAX_CACHE_LEN=max_cache_len,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        HAS_FORCE_FLUSH=force_flush is not None,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, initial_state


# Backwards-compatible aliases: the original GDN-only names. Existing callers
# (backend dispatch, tests, microbench) keep working; ``is_kda`` defaults to
# False so these are the GDN path unchanged.
fused_recurrent_gdn_replayssm_decode_kernel = (
    fused_recurrent_linear_replayssm_decode_kernel
)
fused_recurrent_gdn_replayssm_decode = fused_recurrent_linear_replayssm_decode
