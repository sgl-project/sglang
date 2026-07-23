# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM speculative-decode verify kernel for GDN (Gated DeltaNet).

Ported from the vLLM ReplaySSM reference implementation
(github.com/Johnny-Liou/ReplaySSM, commit ``3c85112``,
``vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py``) and the Dao
AI Lab ReplaySSM blog (dao-lab.ai/blog/2026/replayssm). Part B of SGLang RFC
#28511.

Adapted to SGLang's GDN verify path: the recurrent intermediate-state snapshot
(full ``[V, K]`` state written per draft token to ``intermediate_ssm``) is
replaced by a per-slot **circular cache** of the last ``L`` committed steps'
``(d, k, g)`` records plus a frozen checkpoint ``h0``. The verify output for the
whole draft window is reconstructed *output-only* (never materialising the state)
via the chunked delta-rule ``(I + A)^{-1}`` UT-transform, and the full state is
flushed back only every ``L`` committed tokens. Rejected drafts roll back by a
pointer move on the cursors -- no state write-back.

Closed-loop exact fold (the state / output error split):
  The scheme has two error paths with opposite structure. The OUTPUT is one-shot
  (computed per step, consumed by the sampler, discarded) -- its error never
  compounds. The STATE accumulates: folding the stored ``d`` records open-loop
  (as the vLLM reference does) feeds the chunked transform's cancellation error
  -- amplified up to ``2^(BS-1)`` by ``(I + A)^{-1}`` -- plus their storage
  quantization into every future window, undamped, so the error grows with
  generation length. Therefore the flush here does NOT fold ``d``: instead
  :func:`gdn_replayssm_exact_fold_kernel` sequentially replays the committed
  window from rings of the RAW inputs (``v`` and pre-norm ``k``, both born in
  the activation dtype and hence stored losslessly, plus fp32 ``g`` / ``beta``),
  mirroring ``fused_sigmoid_gating_delta_rule_update_kernel``'s fp32 op order
  exactly. The delta-rule recurrence is contractive (per step the perturbation
  gain is ``exp(g) * |1 - beta| < 1`` along ``k`` and ``exp(g) < 1`` elsewhere),
  so given identical inputs the replayed checkpoint is bit-identical to the
  recurrent baseline's committed state and carries NO length-dependent error.
  The chunked transform is kept only for the non-accumulating output. Its
  dots therefore need only stay below the bf16 OUTPUT-cast floor (eps ~ 2^-8
  ~ 4e-3 relative), not match fp32 exactly: ``DOT_PRECISION`` defaults to
  ``"tf32"`` (~5e-4, tensor-core path; worst case through the (I+A)^{-1}
  amplification 2^(BS-1) still lands at the floor). ``"ieee"`` / ``"tf32x3"``
  remain selectable for ablations. The committed state is untouched by any of
  these dots (requires the fp32 SSM checkpoint; enforced in server_args).

Differences from the vLLM reference:
  * SGLang passes **split** ``q`` / ``k`` / ``v`` tensors (already split + post
    causal-conv1d in the GDN backend) rather than a packed ``mixed_qkv``.
  * State / cache layouts and ``(I + A)^{-1}`` math are unchanged
    (SGLang ``ssm_states`` is ``[slots, HV, V, K]`` with ``K`` contiguous, which
    matches the reference checkpoint exactly).

Linear-chain only: the intra-window interaction uses a strictly-lower causal
mask, so this kernel is valid only for linear draft chains
(``speculative_eagle_topk <= 1``, i.e. MTP / frozen-KV-MTP). Tree verify
(topk > 1), KDA, and NPU/CPU fall back to the recurrent verify kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def gdn_replayssm_spec_circular_kernel(
    q,  # [total_tokens, H, K]
    k,  # [total_tokens, H, K]
    v,  # [total_tokens, HV, V]
    a,  # [total_tokens, HV]
    b,  # [total_tokens, HV]
    A_log,  # [HV] fp32
    dt_bias,  # [HV] fp32
    o,  # [total_tokens, HV, V]  preallocated output
    h0,  # [num_slots, HV, V, K]  checkpoint (read-only; folded by the exact-fold kernel)
    d_cache,  # [num_slots, HV, L, V]  chunked deltas (output reconstruction only)
    k_cache,  # [num_slots, H, L, K]  L2-normalized keys (output reconstruction only)
    g_cache,  # [num_slots, HV, L]  fp32
    rawv_cache,  # [num_slots, HV, L, V]  raw v (exact-fold replay)
    rawk_cache,  # [num_slots, H, L, K]  raw pre-norm k (exact-fold replay)
    beta_cache,  # [num_slots, HV, L]  fp32 beta (exact-fold replay)
    query_start_loc,  # [B+1] int  packed cu_seqlens
    ssm_state_indices,  # [B] int  physical block per request
    write_pos,  # [num_slots] int32  block-keyed
    cache_base,  # [num_slots] int32  block-keyed circular origin
    is_flush_flags,  # [num_slots] int8  block-keyed
    scale,
    stride_q_t: tl.constexpr,  # per-token stride of q (= H*K)
    stride_k_t: tl.constexpr,  # per-token stride of k (= H*K)
    stride_v_t: tl.constexpr,  # per-token stride of v (= HV*V)
    stride_a_t: tl.constexpr,
    stride_b_t: tl.constexpr,
    stride_o_t: tl.constexpr,  # per-token stride of o (= HV*V)
    stride_state_slot: tl.constexpr,
    stride_d_slot: tl.constexpr,
    stride_k_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_qsl: tl.constexpr,
    stride_indices: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr,
    BKT: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_FLUSH: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)

    o_v = i_v * BV + tl.arange(0, BV)
    o_s = tl.arange(0, BS)
    o_c = tl.arange(0, BC)
    mask_v = o_v < V

    # --- per-request packed window ---
    bos = tl.load(query_start_loc + i_n * stride_qsl).to(tl.int64)
    eos = tl.load(query_start_loc + (i_n + 1) * stride_qsl).to(tl.int64)
    spec_len = eos - bos  # full window length
    # Clamp the token index for padding lanes (o_s >= spec_len, which exist when
    # BS = max(bs_min, npow2(max_spec_len)) > spec_len) so masked loads/stores
    # never form out-of-bounds token addresses into the packed varlen buffers.
    # mask_s still zeroes the *values* of those lanes, so this is numerically a
    # no-op -- it only keeps the computed pointers in-bounds.
    mask_s = o_s < spec_len
    o_s_safe = tl.where(mask_s, o_s, 0)

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)

    # output pointer (packed): token (bos + o_s), value-head i_hv, dim o_v
    p_o = o + (bos + o_s_safe[:, None]) * stride_o_t + i_hv * V + o_v[None, :]

    if IS_FLUSH:
        if state_idx <= NULL_BLOCK_ID:
            return
        b_is_flush = tl.load(is_flush_flags + state_idx) != 0
        if not b_is_flush:
            return
    else:
        if state_idx <= NULL_BLOCK_ID:
            full_mask = (o_s < spec_len)[:, None] & mask_v[None, :]
            tl.store(
                p_o,
                tl.zeros([BS, BV], dtype=tl.float32).to(p_o.dtype.element_ty),
                mask=full_mask,
            )
            return
        b_is_flush = tl.load(is_flush_flags + state_idx) != 0
        if b_is_flush:
            return

    b_write_pos = tl.load(write_pos + state_idx).to(tl.int64)
    b_cache_base = tl.load(cache_base + state_idx).to(tl.int32)

    out_mask = mask_s[:, None] & mask_v[None, :]

    b_wp_i = b_write_pos.to(tl.int32)
    cache_valid = o_c < b_write_pos

    # CIRCULAR physical slots (addresses only; masks/cumsums stay logical).
    phys_c = (b_cache_base + o_c) & (MAX_CACHE_LEN - 1)  # [BC] history
    phys_spec = (b_cache_base + b_wp_i + o_s) & (MAX_CACHE_LEN - 1)  # [BS] spec

    # ------------------------------------------------------------------
    # Block 0: gates / beta / local cumsum + committed-history replay decay.
    # ------------------------------------------------------------------
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    a_s = tl.load(a + (bos + o_s_safe) * stride_a_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    b_s = tl.load(b + (bos + o_s_safe) * stride_b_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    x = a_s + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_s = tl.where(mask_s, -tl.exp(A_log_val) * softplus_x, 0.0)
    # Explicit sigmoid expression (NOT tl.sigmoid): bitwise clone of
    # fused_sigmoid_gating_delta_rule_update_kernel's beta so the fp32 value
    # stored to beta_cache is exactly what the recurrent baseline would compute.
    beta_s = tl.where(mask_s, 1.0 / (1.0 + tl.exp(-b_s)), 0.0)
    G_s = tl.cumsum(g_s, axis=0)
    expG_s = tl.exp(G_s)

    if not IS_FLUSH:
        # Committed-history replay decay from cached g (history loads -> phys_c).
        # Output reconstruction only. On flush steps the history is folded into
        # the checkpoint by the exact-fold kernel (launched first), so none of
        # this is needed there.
        p_g_main = g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_c
        b_g_all = tl.load(p_g_main, mask=cache_valid, other=0.0).to(tl.float32)
        b_g_prefix = tl.cumsum(b_g_all, axis=0)
        b_g_total = tl.sum(b_g_all, axis=0)
        b_replay_decay = tl.where(cache_valid, tl.exp(b_g_total - b_g_prefix), 0.0)
        b_total_decay = tl.exp(b_g_total)

        p_d_main = d_cache + (
            state_idx * stride_d_slot
            + (i_hv * MAX_CACHE_LEN + phys_c[None, :]) * V
            + o_v[:, None]
        )
        b_d_all = tl.load(
            p_d_main, mask=mask_v[:, None] & cache_valid[None, :], other=0.0
        ).to(tl.float32)
        b_d_scaled = (b_d_all * b_replay_decay[None, :]).to(h0.dtype.element_ty)

    if USE_QK_L2NORM_IN_KERNEL:
        qnorm_acc = tl.zeros([BS], dtype=tl.float32)
        knorm_acc = tl.zeros([BS], dtype=tl.float32)
        for kk in range(NK):
            o_kt = kk * BKT + tl.arange(0, BKT)
            mask_kt = o_kt < K
            ld = mask_s[:, None] & mask_kt[None, :]
            qn = tl.load(
                q + (bos + o_s_safe[:, None]) * stride_q_t + i_h * K + o_kt[None, :],
                mask=ld,
                other=0.0,
            ).to(tl.float32)
            knn = tl.load(
                k + (bos + o_s_safe[:, None]) * stride_k_t + i_h * K + o_kt[None, :],
                mask=ld,
                other=0.0,
            ).to(tl.float32)
            qnorm_acc += tl.sum(qn * qn, axis=1)
            knorm_acc += tl.sum(knn * knn, axis=1)
        q_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(qnorm_acc + 1e-6), 0.0)
        k_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(knorm_acc + 1e-6), 0.0)
    else:
        q_rnorm = tl.where(mask_s, 1.0, 0.0)
        k_rnorm = tl.where(mask_s, 1.0, 0.0)

    # ------------------------------------------------------------------
    # K-Tiled Fused Projection and Intra-Spec Matrices (+ flush)
    # ------------------------------------------------------------------
    hw_q = tl.zeros([BV, BS], dtype=tl.float32)
    hw_k = tl.zeros([BV, BS], dtype=tl.float32)
    if not IS_FLUSH:
        scores_q = tl.zeros([BC, BS], dtype=tl.float32)
        scores_k = tl.zeros([BC, BS], dtype=tl.float32)
    kk_mat = tl.zeros([BS, BS], dtype=tl.float32)
    kq_mat = tl.zeros([BS, BS], dtype=tl.float32)

    write_k = (i_v == 0) and (i_hv == i_h * (HV // H))

    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        # NOTE: do NOT mask the token (o_s) dimension on these loads. With the
        # o_s_safe clamp the padding-row addresses are in-bounds (they read
        # token `bos`), and `q_rnorm`/`k_rnorm` are already 0 on padding rows
        # (line above), so the multiply below zeroes them. Masking the token
        # dimension here instead leaves the tl.dot shared-memory staging of the
        # padding rows uninitialised -- on this Triton/MMA path that stale
        # staging leaks into the *valid* output columns (manifests at
        # spec_len=2). Loading the full (token-unmasked) tile keeps staging
        # fully written; the result is numerically identical.
        q_tile = tl.load(
            q + (bos + o_s_safe[:, None]) * stride_q_t + i_h * K + o_kt[None, :],
            mask=mask_kt[None, :],
            other=0.0,
        ).to(tl.float32)
        k_tile = tl.load(
            k + (bos + o_s_safe[:, None]) * stride_k_t + i_h * K + o_kt[None, :],
            mask=mask_kt[None, :],
            other=0.0,
        ).to(tl.float32)
        # Keep the raw (pre-norm) key tile for the exact-fold ring: the replay
        # recomputes the L2 norm in fp32 exactly as the recurrent kernel does
        # (its division form differs bitwise from the reciprocal-multiply below).
        k_raw_tile = k_tile
        q_tile = (q_tile * (q_rnorm * scale)[:, None]).to(h0.dtype.element_ty)
        k_tile = (k_tile * k_rnorm[:, None]).to(h0.dtype.element_ty)

        p_h0 = (
            h0
            + state_idx * stride_state_slot
            + i_hv * V * K
            + o_v[:, None] * K
            + o_kt[None, :]
        )
        sc_tile = tl.load(p_h0, mask=mask_v[:, None] & mask_kt[None, :], other=0.0).to(
            h0.dtype.element_ty
        )

        qT = tl.trans(q_tile)
        kT = tl.trans(k_tile)
        kk_mat += tl.dot(k_tile, kT, input_precision=DOT_PRECISION)
        kq_mat += tl.dot(k_tile, qT, input_precision=DOT_PRECISION)

        # Checkpoint projection. On flush steps the exact-fold kernel (launched
        # first) has already folded the committed history into h0, so this is
        # the whole non-window contribution and no d-fold happens here anymore.
        hw_q += tl.dot(sc_tile, qT, input_precision=DOT_PRECISION)
        hw_k += tl.dot(sc_tile, kT, input_precision=DOT_PRECISION)

        if not IS_FLUSH:
            # cached-key history load -> phys_c (output reconstruction only)
            p_k = k_cache + (
                state_idx * stride_k_slot
                + (i_h * MAX_CACHE_LEN + phys_c[:, None]) * K
                + o_kt[None, :]
            )
            khist_tile = tl.load(
                p_k, mask=cache_valid[:, None] & mask_kt[None, :], other=0.0
            ).to(h0.dtype.element_ty)
            scores_q += tl.dot(khist_tile, qT, input_precision=DOT_PRECISION)
            scores_k += tl.dot(khist_tile, kT, input_precision=DOT_PRECISION)

        if write_k:
            spec_kt_mask = (
                mask_s[:, None]
                & mask_kt[None, :]
                & ((b_write_pos + o_s[:, None]) < MAX_CACHE_LEN)
            )
            # raw pre-norm key for the exact-fold replay -> phys_spec (circular).
            # Born in the activation dtype, so the store round-trips losslessly.
            p_cur_rawk = rawk_cache + (
                state_idx * stride_rawk_slot
                + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                + o_kt[None, :]
            )
            tl.store(
                p_cur_rawk,
                k_raw_tile.to(p_cur_rawk.dtype.element_ty),
                mask=spec_kt_mask,
            )
            # normalized spec key store -> phys_spec (circular)
            p_cur_k = k_cache + (
                state_idx * stride_k_slot
                + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                + o_kt[None, :]
            )
            tl.store(
                p_cur_k,
                k_tile,
                mask=spec_kt_mask,
            )

    if not IS_FLUSH:
        hw_q = b_total_decay * hw_q + tl.dot(
            b_d_scaled, scores_q.to(b_d_scaled.dtype), input_precision=DOT_PRECISION
        )
        hw_k = b_total_decay * hw_k + tl.dot(
            b_d_scaled, scores_k.to(b_d_scaled.dtype), input_precision=DOT_PRECISION
        )

    # ------------------------------------------------------------------
    # strictly-lower A and T = (I + A)^{-1}.
    # ------------------------------------------------------------------
    lower = (o_s[:, None] > o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ij = G_s[:, None] - G_s[None, :]
    A_mat = tl.where(lower, beta_s[:, None] * tl.exp(diff_ij) * kk_mat, 0.0)

    b_Ai = -A_mat
    for ii in range(2, BS):
        row = tl.sum(tl.where((o_s == ii)[:, None], -A_mat, 0.0), axis=0)
        row = tl.where(o_s < ii, row, 0.0)
        row = row + tl.sum(row[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_s == ii)[:, None], row, b_Ai)
    T_mat = b_Ai + (o_s[:, None] == o_s[None, :]).to(tl.float32)

    # ------------------------------------------------------------------
    # R and D_spec = R @ T^T.
    # ------------------------------------------------------------------
    p_v = v + (bos + o_s_safe[None, :]) * stride_v_t + i_hv * V + o_v[:, None]
    v_tile = tl.load(p_v, mask=mask_v[:, None] & mask_s[None, :], other=0.0).to(
        tl.float32
    )
    R_mat = beta_s[None, :] * (v_tile - expG_s[None, :] * hw_k)
    D_spec = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Rj = tl.sum(tl.where((o_s == j)[None, :], R_mat, 0.0), axis=1)
        Tj = tl.sum(tl.where((o_s == j)[None, :], T_mat, 0.0), axis=1)
        D_spec += Rj[:, None] * Tj[None, :]

    # ------------------------------------------------------------------
    # outputs.
    # ------------------------------------------------------------------
    causalF = (o_s[:, None] <= o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ji = G_s[None, :] - G_s[:, None]
    F_mat = tl.where(causalF, tl.exp(diff_ji) * kq_mat, 0.0)
    DF = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Dj = tl.sum(tl.where((o_s == j)[None, :], D_spec, 0.0), axis=1)
        Fj = tl.sum(tl.where((o_s == j)[:, None], F_mat, 0.0), axis=0)
        DF += Dj[:, None] * Fj[None, :]
    O_tile = expG_s[None, :] * hw_q + DF

    tl.store(p_o, tl.trans(O_tile).to(p_o.dtype.element_ty), mask=out_mask)

    # ------------------------------------------------------------------
    # write speculative d / raw-v / g / beta at circular positions (phys_spec).
    # ------------------------------------------------------------------
    spec_pos = b_write_pos + o_s
    spec_store_mask = mask_s & (spec_pos < MAX_CACHE_LEN)
    p_cur_d = d_cache + (
        state_idx * stride_d_slot
        + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
        + o_v[:, None]
    )
    tl.store(
        p_cur_d,
        D_spec.to(p_cur_d.dtype.element_ty),
        mask=mask_v[:, None] & spec_store_mask[None, :],
    )
    # raw v for the exact-fold replay (activation dtype: lossless round-trip)
    p_cur_v = rawv_cache + (
        state_idx * stride_rawv_slot
        + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
        + o_v[:, None]
    )
    tl.store(
        p_cur_v,
        v_tile.to(p_cur_v.dtype.element_ty),
        mask=mask_v[:, None] & spec_store_mask[None, :],
    )
    if i_v == 0:
        p_cur_g = g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_spec
        tl.store(p_cur_g, g_s, mask=spec_store_mask)
        p_cur_beta = (
            beta_cache + state_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN + phys_spec
        )
        tl.store(p_cur_beta, beta_s, mask=spec_store_mask)


@triton.jit
def gdn_replayssm_exact_fold_kernel(
    h0,  # [num_slots, HV, V, K]  fp32 checkpoint (in-place)
    rawv_cache,  # [num_slots, HV, L, V]  raw v
    rawk_cache,  # [num_slots, H, L, K]  raw pre-norm k
    g_cache,  # [num_slots, HV, L]  fp32
    beta_cache,  # [num_slots, HV, L]  fp32
    ssm_state_indices,  # [B] int  physical block per request
    write_pos,  # [num_slots] int32  block-keyed
    cache_base,  # [num_slots] int32  block-keyed
    is_flush_flags,  # [num_slots] int8  block-keyed
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_indices: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
):
    """Closed-loop exact fold: sequentially replay the committed ring window
    into the fp32 checkpoint on flush rows.

    BITWISE CLONE of ``fused_sigmoid_gating_delta_rule_update_kernel``'s state
    recurrence (fused_sigmoid_gating_recurrent.py) -- same [BK, BV] register
    tile with the K axis as rows, same 1-D full-K reductions, the same
    division-form L2 norm with eps inside the sqrt, and the same
    decay -> delta -> rank-1-update op order. Given identical inputs the
    committed state matches the recurrent baseline bit-for-bit, which is the
    whole point: do NOT "optimize" this into tl.dot / reciprocal-multiply /
    reordered expressions, and keep the launch config (BV=32, num_warps=1)
    identical to the recurrent kernel so the reduction trees agree.

    Replaces the previous open-loop d-fold, whose per-flush error (chunked
    (I+A)^{-1} cancellation + d-storage quantization) fed forward across
    flushes undamped and grew with generation length.
    """
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    if tl.load(is_flush_flags + state_idx) == 0:
        return
    b_write_pos = tl.load(write_pos + state_idx).to(tl.int32)
    if b_write_pos <= 0:
        return
    b_cache_base = tl.load(cache_base + state_idx).to(tl.int32)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # [BK, BV] tile, K rows / V columns -- the recurrent kernel's layout
    # (memory offset v * K + k, K contiguous).
    p_h0 = (
        h0
        + state_idx * stride_state_slot
        + i_hv * V * K
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for t in range(0, b_write_pos):
        phys = ((b_cache_base + t) & (MAX_CACHE_LEN - 1)).to(tl.int64)
        b_k = tl.load(
            rawk_cache
            + state_idx * stride_rawk_slot
            + (i_h * MAX_CACHE_LEN + phys) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            rawv_cache
            + state_idx * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + phys) * V
            + o_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        b_g = tl.load(
            g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys
        ).to(tl.float32)
        b_beta = tl.load(
            beta_cache + state_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN + phys
        ).to(tl.float32)

        # --- verbatim recurrent update (see the clone note above) ---
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@triton.jit
def _advance_gdn_spec_cursors_kernel(
    write_pos_ptr,
    cache_base_ptr,
    is_flush_ptr,
    num_accepted_ptr,
    state_batch_indices_ptr,
    n_rows,
    stride_sbi: tl.constexpr,
    stride_na: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
    CACHE_BUF_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    row_mask = offs < n_rows
    blk = tl.load(
        state_batch_indices_ptr + offs * stride_sbi, mask=row_mask, other=NULL_BLOCK_ID
    ).to(tl.int64)
    valid = row_mask & (blk > NULL_BLOCK_ID)

    write_pos = tl.load(write_pos_ptr + blk, mask=valid, other=0).to(tl.int32)
    cache_base = tl.load(cache_base_ptr + blk, mask=valid, other=0).to(tl.int32)
    is_flush_cur = tl.load(is_flush_ptr + blk, mask=valid, other=0).to(tl.int32)
    num_acc = tl.load(num_accepted_ptr + offs * stride_na, mask=valid, other=0).to(
        tl.int32
    )

    total_commit = num_acc
    flush_now = (total_commit > 0) & (is_flush_cur != 0)

    new_base = tl.where(
        flush_now, (cache_base + write_pos) & (CACHE_BUF_LEN - 1), cache_base
    )
    new_wp = tl.where(is_flush_cur != 0, total_commit, write_pos + total_commit).to(
        tl.int32
    )
    # EARLY-FLUSH (margin = 2 * max_spec_len, strict '>'): flush one window early
    # so that on every verify step write_pos + spec_len <= max_cache_len holds,
    # i.e. the spec window NEVER overflows the circular cache. This is required
    # for e2e correctness: the proposer (n-gram / MTP) cannot be told to cap its
    # draft count, and the rejection sampler reads logits for EVERY window
    # position -- so an overflowing position would feed the sampler an
    # uninitialized-`out` garbage logit (emitting a wrong token) and desync the
    # committed state. The strict '>' uses the buffer exactly (max write_pos at a
    # flush step = max_cache_len - max_spec_len, zero headroom); usable committed
    # history is max_cache_len - 2*max_spec_len + 1; raise the cache length
    # for more. Config enforces max_cache_len >= 2 * max_spec_len.
    next_is_flush = ((new_wp + 2 * MAX_SPEC_LEN) > MAX_CACHE_LEN).to(tl.int8)

    tl.store(write_pos_ptr + blk, new_wp, mask=valid)
    tl.store(cache_base_ptr + blk, new_base, mask=valid)
    tl.store(is_flush_ptr + blk, next_is_flush, mask=valid)


@triton.jit
def _reset_gdn_replayssm_spec_cursors_kernel(
    write_pos_ptr,
    cache_base_ptr,
    is_flush_ptr,
    do_reset_ptr,
    state_batch_indices_ptr,
    n_rows,
    stride_sbi: tl.constexpr,
    stride_reset: tl.constexpr,
    INIT_FLUSH: tl.constexpr,
    BLOCK: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    row_mask = offs < n_rows
    blk = tl.load(
        state_batch_indices_ptr + offs * stride_sbi, mask=row_mask, other=NULL_BLOCK_ID
    ).to(tl.int64)
    do_reset = tl.load(do_reset_ptr + offs * stride_reset, mask=row_mask, other=0).to(
        tl.int32
    )
    do = row_mask & (blk > NULL_BLOCK_ID) & (do_reset != 0)

    tl.store(write_pos_ptr + blk, tl.zeros_like(blk).to(tl.int32), mask=do)
    tl.store(cache_base_ptr + blk, tl.zeros_like(blk).to(tl.int32), mask=do)
    tl.store(
        is_flush_ptr + blk,
        tl.full([BLOCK], INIT_FLUSH, dtype=tl.int8),
        mask=do,
    )


# ---------------------------------------------------------------------------
# Python wrappers.
# ---------------------------------------------------------------------------
def _launch_gdn_spec(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    out,
    checkpoint_state,
    d_cache,
    k_cache,
    g_cache,
    rawv_cache,
    rawk_cache,
    beta_cache,
    query_start_loc,
    ssm_state_indices,
    write_pos,
    cache_base,
    is_flush,
    scale,
    max_cache_len,
    max_spec_len,
    use_qk_l2norm_in_kernel,
    is_flush_kernel,
    block_v,
    num_warps,
    num_stages,
    nk,
    bs_min,
    null_block_id,
    dot_precision,
):
    num_slots, HV, V, K = checkpoint_state.shape
    H = k.shape[1]
    B = query_start_loc.shape[0] - 1
    assert (
        max_cache_len & (max_cache_len - 1) == 0
    ), "circular cache requires power-of-two max_cache_len"
    assert d_cache.shape[2] == max_cache_len

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(f"only NK_global=1 supported (K={K}, BK={BK}).")
    if BK % nk != 0:
        raise ValueError(f"nk={nk} must divide BK={BK}.")
    BKT = BK // nk
    if BKT < 16:
        raise ValueError(f"BKT={BKT} must be >=16 for tl.dot.")
    BV = block_v if block_v is not None else min(triton.next_power_of_2(V), 64)
    BS = max(bs_min, triton.next_power_of_2(max_spec_len))
    BC = max(16, triton.next_power_of_2(max_cache_len))

    grid = (triton.cdiv(V, BV), B, HV)
    gdn_replayssm_spec_circular_kernel[grid](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        out,
        checkpoint_state,
        d_cache,
        k_cache,
        g_cache,
        rawv_cache,
        rawk_cache,
        beta_cache,
        query_start_loc,
        ssm_state_indices,
        write_pos,
        cache_base,
        is_flush,
        scale,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        a.stride(0),
        b.stride(0),
        out.stride(0),
        checkpoint_state.stride(0),
        d_cache.stride(0),
        k_cache.stride(0),
        g_cache.stride(0),
        rawv_cache.stride(0),
        rawk_cache.stride(0),
        beta_cache.stride(0),
        query_start_loc.stride(0),
        ssm_state_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BS=BS,
        BC=BC,
        NK=nk,
        BKT=BKT,
        MAX_CACHE_LEN=max_cache_len,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_FLUSH=is_flush_kernel,
        NULL_BLOCK_ID=null_block_id,
        DOT_PRECISION=dot_precision,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_gdn_exact_fold(
    checkpoint_state,
    rawv_cache,
    rawk_cache,
    g_cache,
    beta_cache,
    query_start_loc,
    ssm_state_indices,
    write_pos,
    cache_base,
    is_flush,
    max_cache_len,
    use_qk_l2norm_in_kernel,
    null_block_id,
    num_k_heads,
):
    """Launch the closed-loop exact fold (flush rows only; device-routed).

    Tiling clones the recurrent kernel exactly (full-K rows, BV = min(np2(V), 32)
    columns, num_warps=1, num_stages=3) so every reduction tree matches
    ``fused_sigmoid_gating_delta_rule_update`` and the folded checkpoint is
    bit-identical to the recurrent baseline's committed state.
    """
    num_slots, HV, V, K = checkpoint_state.shape
    B = query_start_loc.shape[0] - 1
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    grid = (triton.cdiv(V, BV), B, HV)
    gdn_replayssm_exact_fold_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        g_cache,
        beta_cache,
        ssm_state_indices,
        write_pos,
        cache_base,
        is_flush,
        checkpoint_state.stride(0),
        rawv_cache.stride(0),
        rawk_cache.stride(0),
        g_cache.stride(0),
        beta_cache.stride(0),
        ssm_state_indices.stride(0),
        H=num_k_heads,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        MAX_CACHE_LEN=max_cache_len,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        NULL_BLOCK_ID=null_block_id,
        num_warps=1,
        num_stages=3,
    )


def gdn_replayssm_spec_decode(
    q: torch.Tensor,  # [total_tokens, H, K]  post-conv
    k: torch.Tensor,  # [total_tokens, H, K]  post-conv
    v: torch.Tensor,  # [total_tokens, HV, V] post-conv
    a: torch.Tensor,  # [total_tokens, HV]
    b: torch.Tensor,  # [total_tokens, HV]
    A_log: torch.Tensor,  # [HV] fp32
    dt_bias: torch.Tensor,  # [HV] fp32
    checkpoint_state: torch.Tensor,  # [num_slots, HV, V, K]  fp32 (folded in-place)
    d_cache: torch.Tensor,  # [num_slots, HV, L, V]
    k_cache: torch.Tensor,  # [num_slots, H, L, K]
    g_cache: torch.Tensor,  # [num_slots, HV, L]  fp32
    rawv_cache: torch.Tensor,  # [num_slots, HV, L, V]  raw v (exact fold)
    rawk_cache: torch.Tensor,  # [num_slots, H, L, K]  raw pre-norm k (exact fold)
    beta_cache: torch.Tensor,  # [num_slots, HV, L]  fp32 beta (exact fold)
    out: torch.Tensor,  # [total_tokens, HV, V]  preallocated
    query_start_loc: torch.Tensor,  # [B+1] int
    ssm_state_indices: torch.Tensor,  # [B] int  physical block per request
    write_pos: torch.Tensor,  # [num_slots] int32  block-keyed
    cache_base: torch.Tensor,  # [num_slots] int32  block-keyed
    is_flush: torch.Tensor,  # [num_slots] int8  block-keyed
    max_cache_len: int,
    max_spec_len: int,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = 0,
    block_v: int = 64,
    num_warps: int = 1,
    num_stages: int = 2,
    nk: int = 2,
    bs_min: int = 4,
    block_v_flush: int = 64,
    num_warps_flush: int = 1,
    num_stages_flush: int = 2,
    nk_flush: int = 2,
    launch_mode: str = "both",
    dot_precision: str = "tf32",
):
    """GDN cached speculative-decode on a CIRCULAR ring cache (split-qkv varlen).

    Three launches, all device-routed per row so the step stays CUDA-graph
    capturable:
      1. exact fold (flush rows): sequentially replay the committed ring window
         (raw v / raw k / g / beta) into the fp32 checkpoint, bit-identical to
         the recurrent baseline (closed loop -- no accumulating state error);
      2. verify (non-flush rows): chunked output from checkpoint + d/k history;
      3. flush-output (flush rows): chunked output from the freshly folded
         checkpoint (statically empty history).
    Cursors are block-keyed (indexed by ``ssm_state_indices``) and advanced
    out-of-kernel by :func:`commit_gdn_replayssm_spec`.
    """
    if scale is None:
        scale = checkpoint_state.shape[-1] ** -0.5
    if is_flush.dtype != torch.int8:
        is_flush = is_flush.to(torch.int8)

    if launch_mode in ("both", "flush"):
        # Must precede the flush-output launch: it reads the folded checkpoint.
        _launch_gdn_exact_fold(
            checkpoint_state,
            rawv_cache,
            rawk_cache,
            g_cache,
            beta_cache,
            query_start_loc,
            ssm_state_indices,
            write_pos,
            cache_base,
            is_flush,
            max_cache_len,
            use_qk_l2norm_in_kernel,
            null_block_id,
            k.shape[1],
        )
    if launch_mode in ("both", "verify"):
        _launch_gdn_spec(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            out,
            checkpoint_state,
            d_cache,
            k_cache,
            g_cache,
            rawv_cache,
            rawk_cache,
            beta_cache,
            query_start_loc,
            ssm_state_indices,
            write_pos,
            cache_base,
            is_flush,
            scale,
            max_cache_len,
            max_spec_len,
            use_qk_l2norm_in_kernel,
            False,
            block_v,
            num_warps,
            num_stages,
            nk,
            bs_min,
            null_block_id,
            dot_precision,
        )
    if launch_mode in ("both", "flush"):
        _launch_gdn_spec(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            out,
            checkpoint_state,
            d_cache,
            k_cache,
            g_cache,
            rawv_cache,
            rawk_cache,
            beta_cache,
            query_start_loc,
            ssm_state_indices,
            write_pos,
            cache_base,
            is_flush,
            scale,
            max_cache_len,
            max_spec_len,
            use_qk_l2norm_in_kernel,
            True,
            block_v_flush,
            num_warps_flush,
            num_stages_flush,
            nk_flush,
            bs_min,
            null_block_id,
            dot_precision,
        )
    return out


def commit_gdn_replayssm_spec(
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    num_accepted: torch.Tensor,  # [n_rows] int  (already includes the bonus token)
    state_batch_indices: torch.Tensor,  # [n_rows] int  physical block per row
    max_cache_len: int,
    max_spec_len: int,
    cache_buf_len: int | None = None,
    null_block_id: int = 0,
):
    """Advance the block-keyed cursors once per decode step (device-only)."""
    if cache_buf_len is None:
        cache_buf_len = max_cache_len
    n_rows = state_batch_indices.shape[0]
    BLOCK = triton.next_power_of_2(max(1, n_rows))
    _advance_gdn_spec_cursors_kernel[(1,)](
        write_pos,
        cache_base,
        is_flush,
        num_accepted,
        state_batch_indices,
        n_rows,
        stride_sbi=state_batch_indices.stride(0),
        stride_na=num_accepted.stride(0),
        MAX_CACHE_LEN=max_cache_len,
        MAX_SPEC_LEN=max_spec_len,
        CACHE_BUF_LEN=cache_buf_len,
        BLOCK=BLOCK,
        NULL_BLOCK_ID=null_block_id,
    )


def reset_gdn_replayssm_spec_cursors(
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    do_reset: torch.Tensor,  # [n_rows] int/bool  1 for first-decode rows
    state_batch_indices: torch.Tensor,  # [n_rows] int
    max_cache_len: int,
    max_spec_len: int,
    null_block_id: int = 0,
):
    """Reset the cursors of first-decode rows (prefill->decode handoff)."""
    n_rows = state_batch_indices.shape[0]
    BLOCK = triton.next_power_of_2(max(1, n_rows))
    # Early-flush margin (2 * max_spec_len, strict '>'): mirror _advance_gdn_spec_cursors_kernel.
    init_flush = 1 if 2 * max_spec_len > max_cache_len else 0
    _reset_gdn_replayssm_spec_cursors_kernel[(1,)](
        write_pos,
        cache_base,
        is_flush,
        do_reset,
        state_batch_indices,
        n_rows,
        stride_sbi=state_batch_indices.stride(0),
        stride_reset=do_reset.stride(0),
        INIT_FLUSH=init_flush,
        BLOCK=BLOCK,
        NULL_BLOCK_ID=null_block_id,
    )
