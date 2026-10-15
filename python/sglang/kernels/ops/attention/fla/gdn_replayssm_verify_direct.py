# SPDX-License-Identifier: Apache-2.0
# Modified by NVIDIA: forward substitution and fused output accumulation.

import triton
import triton.language as tl


@triton.jit
def _gdn_replayssm_verify_body(
    q,  # [total_tokens, H, K]
    k,  # [total_tokens, H, K]
    v,  # [total_tokens, HV, V]
    a,  # [total_tokens, HV]
    b,  # [total_tokens, HV]
    A_log,  # [HV] fp32
    dt_bias,  # [HV] fp32
    o,  # [total_tokens, HV, V]  preallocated output
    h0,  # [num_slots, HV, V, K]  checkpoint (read-only in verify)
    d_cache,  # [num_slots, HV, L, V]  chunked deltas (output reconstruction only)
    k_cache,  # [num_slots, H, L, K]  L2-normalized keys (output reconstruction only)
    g_cache,  # [num_slots, HV, L]  fp32
    rawv_cache,  # exact raw v or compact D low part
    rawk_cache,  # exact raw k or normalized-K low part
    beta_cache,  # fp32 beta for exact replay
    query_start_loc,  # [B+1] int  packed cu_seqlens
    ssm_state_indices,  # [B] int  physical block per request
    replay_indices,  # [B] int  active request slot per request
    write_pos,  # [num_replay_slots] int32
    cache_base,  # [num_replay_slots] int32
    is_flush_flags,  # [num_replay_slots] int8
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
    stride_replay_indices: tl.constexpr,
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
    STATE_FP32: tl.constexpr,
    STORE_EXACT_RAW: tl.constexpr,
    STORE_D_RESIDUAL: tl.constexpr,
    STORE_K_RESIDUAL: tl.constexpr,
    IS_FLUSH: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    DIRECT_SOLVE: tl.constexpr = False,
    SPEC_TOKENS: tl.constexpr = 0,
    EMPTY_HISTORY: tl.constexpr = False,
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
    replay_idx = tl.load(replay_indices + i_n * stride_replay_indices).to(tl.int64)

    # output pointer (packed): token (bos + o_s), value-head i_hv, dim o_v
    p_o = o + (bos + o_s_safe[:, None]) * stride_o_t + i_hv * V + o_v[None, :]

    if IS_FLUSH:
        if state_idx <= NULL_BLOCK_ID:
            return
        b_is_flush = tl.load(is_flush_flags + replay_idx) != 0
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
        b_is_flush = tl.load(is_flush_flags + replay_idx) != 0
        if b_is_flush:
            return

    b_write_pos = tl.load(write_pos + replay_idx).to(tl.int64)
    b_cache_base = tl.load(cache_base + replay_idx).to(tl.int32)

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

    read_history: tl.constexpr = (not IS_FLUSH) and (not EMPTY_HISTORY)
    if read_history:
        # Committed-history replay decay from cached g (history loads -> phys_c).
        # Output reconstruction only. On flush steps the history is folded into
        # the checkpoint by the exact-fold kernel (launched first), so none of
        # this is needed there.
        p_g_main = g_cache + replay_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_c
        b_g_all = tl.load(p_g_main, mask=cache_valid, other=0.0).to(tl.float32)
        b_g_prefix = tl.cumsum(b_g_all, axis=0)
        b_g_total = tl.sum(b_g_all, axis=0)
        b_replay_decay = tl.where(cache_valid, tl.exp(b_g_total - b_g_prefix), 0.0)
        b_total_decay = tl.exp(b_g_total)

        p_d_main = d_cache + (
            replay_idx * stride_d_slot
            + (i_hv * MAX_CACHE_LEN + phys_c[None, :]) * V
            + o_v[:, None]
        )
        b_d_all = tl.load(
            p_d_main, mask=mask_v[:, None] & cache_valid[None, :], other=0.0
        ).to(tl.float32)
        b_d_scaled_fp32 = b_d_all * b_replay_decay[None, :]
        if STATE_FP32:
            # Preserve the pre-existing fp32-checkpoint path: these dots use
            # fp32 operands with DOT_PRECISION (TF32 by default).
            b_d_scaled = b_d_scaled_fp32
        else:
            b_d_scaled = b_d_scaled_fp32.to(d_cache.dtype.element_ty)

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
        k_raw_tile = k_tile
        q_precise = q_tile * (q_rnorm * scale)[:, None]
        k_precise = k_tile * k_rnorm[:, None]
        k_store_tile = k_precise.to(k_cache.dtype.element_ty)
        k_residual_tile = (k_precise - k_store_tile.to(tl.float32)).to(
            k.dtype.element_ty
        )
        if STATE_FP32:
            # Do not lower the fp32-checkpoint output path to activation dtype.
            q_tile = q_precise
            k_tile = k_precise
        else:
            q_tile = q_precise.to(q.dtype.element_ty)
            k_tile = k_store_tile

        p_h0 = (
            h0
            + state_idx * stride_state_slot
            + i_hv * V * K
            + o_v[:, None] * K
            + o_kt[None, :]
        )
        sc_tile = tl.load(p_h0, mask=mask_v[:, None] & mask_kt[None, :], other=0.0)

        qT = tl.trans(q_tile)
        kT = tl.trans(k_tile)
        kk_mat += tl.dot(k_tile, kT, input_precision=DOT_PRECISION)
        kq_mat += tl.dot(k_tile, qT, input_precision=DOT_PRECISION)
        if STORE_K_RESIDUAL and not STATE_FP32:
            k_residual_T = tl.trans(k_residual_tile)
            kk_mat += tl.dot(k_tile, k_residual_T, input_precision=DOT_PRECISION)
            kk_mat += tl.dot(k_residual_tile, kT, input_precision=DOT_PRECISION)
            kk_mat += tl.dot(
                k_residual_tile, k_residual_T, input_precision=DOT_PRECISION
            )
            kq_mat += tl.dot(k_residual_tile, qT, input_precision=DOT_PRECISION)

        # Checkpoint projection. On flush steps the exact-fold kernel (launched
        # first) has already folded the committed history into h0, so this is
        # the whole non-window contribution and no d-fold happens here anymore.
        if STATE_FP32:
            sc_tile = sc_tile.to(tl.float32)
            hw_q += tl.dot(sc_tile, qT, input_precision=DOT_PRECISION)
            hw_k += tl.dot(sc_tile, kT, input_precision=DOT_PRECISION)
        else:
            sc_tile = sc_tile.to(k_tile.dtype)
            hw_q += tl.dot(sc_tile, qT)
            hw_k += tl.dot(sc_tile, kT)
            if STORE_K_RESIDUAL:
                hw_k += tl.dot(sc_tile, k_residual_T)

        if read_history:
            # cached-key history load -> phys_c (output reconstruction only)
            p_k = k_cache + (
                replay_idx * stride_k_slot
                + (i_h * MAX_CACHE_LEN + phys_c[:, None]) * K
                + o_kt[None, :]
            )
            khist_tile = tl.load(
                p_k, mask=cache_valid[:, None] & mask_kt[None, :], other=0.0
            )
            if STATE_FP32:
                khist_tile = khist_tile.to(tl.float32)
            else:
                khist_tile = khist_tile.to(k_cache.dtype.element_ty)
            scores_q += tl.dot(khist_tile, qT, input_precision=DOT_PRECISION)
            scores_k += tl.dot(khist_tile, kT, input_precision=DOT_PRECISION)

        if write_k:
            spec_kt_mask = (
                mask_s[:, None]
                & mask_kt[None, :]
                & ((b_write_pos + o_s[:, None]) < MAX_CACHE_LEN)
            )
            if STORE_EXACT_RAW or STORE_K_RESIDUAL:
                p_cur_rawk = rawk_cache + (
                    replay_idx * stride_rawk_slot
                    + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                    + o_kt[None, :]
                )
                if STORE_EXACT_RAW:
                    tl.store(
                        p_cur_rawk,
                        k_raw_tile.to(p_cur_rawk.dtype.element_ty),
                        mask=spec_kt_mask,
                    )
                else:
                    tl.store(p_cur_rawk, k_residual_tile, mask=spec_kt_mask)
            # normalized spec key store -> phys_spec (circular)
            p_cur_k = k_cache + (
                replay_idx * stride_k_slot
                + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                + o_kt[None, :]
            )
            tl.store(
                p_cur_k,
                k_store_tile,
                mask=spec_kt_mask,
            )

    if read_history:
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
    decay_mat = tl.exp(diff_ij)
    A_mat = tl.where(lower, beta_s[:, None] * decay_mat * kk_mat, 0.0)

    if not DIRECT_SOLVE:
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
    causalF = (o_s[:, None] <= o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    F_mat = tl.where(causalF, tl.trans(decay_mat) * kq_mat, 0.0)
    DF = tl.zeros([BV, BS], dtype=tl.float32)
    if DIRECT_SOLVE:
        # Solve D (I + A)^T = R in token order without forming the inverse.
        # Keep the compact-cache rounding below identical to the other path.
        for j in tl.static_range(SPEC_TOKENS if SPEC_TOKENS else BS):
            Dj = tl.sum(tl.where((o_s == j)[None, :], R_mat, 0.0), axis=1)
            D_spec = tl.where((o_s == j)[None, :], Dj[:, None], D_spec)
            Aj = tl.sum(tl.where((o_s == j)[None, :], A_mat, 0.0), axis=1)
            R_mat -= Dj[:, None] * Aj[None, :]
            Fj = tl.sum(tl.where((o_s == j)[:, None], F_mat, 0.0), axis=0)
            DF += Dj[:, None] * Fj[None, :]
    else:
        for j in tl.static_range(BS):
            Rj = tl.sum(tl.where((o_s == j)[None, :], R_mat, 0.0), axis=1)
            Tj = tl.sum(tl.where((o_s == j)[None, :], T_mat, 0.0), axis=1)
            D_spec += Rj[:, None] * Tj[None, :]

    # ------------------------------------------------------------------
    # outputs.
    # ------------------------------------------------------------------
    if not DIRECT_SOLVE:
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
        replay_idx * stride_d_slot
        + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
        + o_v[:, None]
    )
    d_hi = D_spec.to(p_cur_d.dtype.element_ty)
    tl.store(
        p_cur_d,
        d_hi,
        mask=mask_v[:, None] & spec_store_mask[None, :],
    )
    if STORE_EXACT_RAW or STORE_D_RESIDUAL:
        p_cur_v = rawv_cache + (
            replay_idx * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
            + o_v[:, None]
        )
        if STORE_EXACT_RAW:
            tl.store(
                p_cur_v,
                v_tile.to(p_cur_v.dtype.element_ty),
                mask=mask_v[:, None] & spec_store_mask[None, :],
            )
        else:
            tl.store(
                p_cur_v,
                (D_spec - d_hi.to(tl.float32)).to(p_cur_v.dtype.element_ty),
                mask=mask_v[:, None] & spec_store_mask[None, :],
            )
    if i_v == 0:
        p_cur_g = (
            g_cache + replay_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_spec
        )
        tl.store(p_cur_g, g_s, mask=spec_store_mask)
        if STORE_EXACT_RAW:
            p_cur_beta = (
                beta_cache
                + replay_idx * stride_beta_slot
                + i_hv * MAX_CACHE_LEN
                + phys_spec
            )
            tl.store(p_cur_beta, beta_s, mask=spec_store_mask)


@triton.jit
def gdn_replayssm_verify_direct_kernel(
    q,  # [total_tokens, H, K]
    k,  # [total_tokens, H, K]
    v,  # [total_tokens, HV, V]
    a,  # [total_tokens, HV]
    b,  # [total_tokens, HV]
    A_log,  # [HV] fp32
    dt_bias,  # [HV] fp32
    o,  # [total_tokens, HV, V]  preallocated output
    h0,  # [num_slots, HV, V, K]  checkpoint (read-only in verify)
    d_cache,  # [num_slots, HV, L, V]  chunked deltas (output reconstruction only)
    k_cache,  # [num_slots, H, L, K]  L2-normalized keys (output reconstruction only)
    g_cache,  # [num_slots, HV, L]  fp32
    rawv_cache,  # exact raw v or compact D low part
    rawk_cache,  # exact raw k or normalized-K low part
    beta_cache,  # fp32 beta for exact replay
    query_start_loc,  # [B+1] int  packed cu_seqlens
    ssm_state_indices,  # [B] int  physical block per request
    replay_indices,  # [B] int  active request slot per request
    write_pos,  # [num_replay_slots] int32
    cache_base,  # [num_replay_slots] int32
    is_flush_flags,  # [num_replay_slots] int8
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
    stride_replay_indices: tl.constexpr,
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
    STATE_FP32: tl.constexpr,
    STORE_EXACT_RAW: tl.constexpr,
    STORE_D_RESIDUAL: tl.constexpr,
    STORE_K_RESIDUAL: tl.constexpr,
    IS_FLUSH: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    DIRECT_SOLVE: tl.constexpr = False,
    SPEC_TOKENS: tl.constexpr = 0,
    CHECK_EMPTY_HISTORY: tl.constexpr = False,
    HISTORY_STATS=None,
    USE_GDC: tl.constexpr = False,
):
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
        # Counters and compact-history writes remain after the prerequisite.
        tl.extra.cuda.gdc_wait()
    if CHECK_EMPTY_HISTORY:
        row = tl.program_id(1)
        replay = tl.load(replay_indices + row * stride_replay_indices).to(tl.int64)
        state = tl.load(ssm_state_indices + row * stride_indices).to(tl.int64)
        valid_state = state > NULL_BLOCK_ID
        cursor = tl.load(write_pos + replay, mask=valid_state, other=0)
        flush = tl.load(is_flush_flags + replay, mask=valid_state, other=0) != 0
        active = valid_state & (flush == IS_FLUSH)
        if tl.program_id(0) == 0 and tl.program_id(2) == 0 and active:
            tl.atomic_add(HISTORY_STATS + tl.where(cursor == 0, 0, 1), 1, sem="relaxed")
        if cursor == 0:
            _gdn_replayssm_verify_body(
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                o=o,
                h0=h0,
                d_cache=d_cache,
                k_cache=k_cache,
                g_cache=g_cache,
                rawv_cache=rawv_cache,
                rawk_cache=rawk_cache,
                beta_cache=beta_cache,
                query_start_loc=query_start_loc,
                ssm_state_indices=ssm_state_indices,
                replay_indices=replay_indices,
                write_pos=write_pos,
                cache_base=cache_base,
                is_flush_flags=is_flush_flags,
                scale=scale,
                stride_q_t=stride_q_t,
                stride_k_t=stride_k_t,
                stride_v_t=stride_v_t,
                stride_a_t=stride_a_t,
                stride_b_t=stride_b_t,
                stride_o_t=stride_o_t,
                stride_state_slot=stride_state_slot,
                stride_d_slot=stride_d_slot,
                stride_k_slot=stride_k_slot,
                stride_g_slot=stride_g_slot,
                stride_rawv_slot=stride_rawv_slot,
                stride_rawk_slot=stride_rawk_slot,
                stride_beta_slot=stride_beta_slot,
                stride_qsl=stride_qsl,
                stride_indices=stride_indices,
                stride_replay_indices=stride_replay_indices,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                BS=BS,
                BC=BC,
                NK=NK,
                BKT=BKT,
                MAX_CACHE_LEN=MAX_CACHE_LEN,
                SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
                USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
                STATE_FP32=STATE_FP32,
                STORE_EXACT_RAW=STORE_EXACT_RAW,
                STORE_D_RESIDUAL=STORE_D_RESIDUAL,
                STORE_K_RESIDUAL=STORE_K_RESIDUAL,
                IS_FLUSH=IS_FLUSH,
                NULL_BLOCK_ID=NULL_BLOCK_ID,
                DOT_PRECISION=DOT_PRECISION,
                DIRECT_SOLVE=DIRECT_SOLVE,
                SPEC_TOKENS=SPEC_TOKENS,
                EMPTY_HISTORY=True,
            )
        else:
            _gdn_replayssm_verify_body(
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                o=o,
                h0=h0,
                d_cache=d_cache,
                k_cache=k_cache,
                g_cache=g_cache,
                rawv_cache=rawv_cache,
                rawk_cache=rawk_cache,
                beta_cache=beta_cache,
                query_start_loc=query_start_loc,
                ssm_state_indices=ssm_state_indices,
                replay_indices=replay_indices,
                write_pos=write_pos,
                cache_base=cache_base,
                is_flush_flags=is_flush_flags,
                scale=scale,
                stride_q_t=stride_q_t,
                stride_k_t=stride_k_t,
                stride_v_t=stride_v_t,
                stride_a_t=stride_a_t,
                stride_b_t=stride_b_t,
                stride_o_t=stride_o_t,
                stride_state_slot=stride_state_slot,
                stride_d_slot=stride_d_slot,
                stride_k_slot=stride_k_slot,
                stride_g_slot=stride_g_slot,
                stride_rawv_slot=stride_rawv_slot,
                stride_rawk_slot=stride_rawk_slot,
                stride_beta_slot=stride_beta_slot,
                stride_qsl=stride_qsl,
                stride_indices=stride_indices,
                stride_replay_indices=stride_replay_indices,
                H=H,
                HV=HV,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                BS=BS,
                BC=BC,
                NK=NK,
                BKT=BKT,
                MAX_CACHE_LEN=MAX_CACHE_LEN,
                SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
                USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
                STATE_FP32=STATE_FP32,
                STORE_EXACT_RAW=STORE_EXACT_RAW,
                STORE_D_RESIDUAL=STORE_D_RESIDUAL,
                STORE_K_RESIDUAL=STORE_K_RESIDUAL,
                IS_FLUSH=IS_FLUSH,
                NULL_BLOCK_ID=NULL_BLOCK_ID,
                DOT_PRECISION=DOT_PRECISION,
                DIRECT_SOLVE=DIRECT_SOLVE,
                SPEC_TOKENS=SPEC_TOKENS,
                EMPTY_HISTORY=False,
            )
    else:
        _gdn_replayssm_verify_body(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            o=o,
            h0=h0,
            d_cache=d_cache,
            k_cache=k_cache,
            g_cache=g_cache,
            rawv_cache=rawv_cache,
            rawk_cache=rawk_cache,
            beta_cache=beta_cache,
            query_start_loc=query_start_loc,
            ssm_state_indices=ssm_state_indices,
            replay_indices=replay_indices,
            write_pos=write_pos,
            cache_base=cache_base,
            is_flush_flags=is_flush_flags,
            scale=scale,
            stride_q_t=stride_q_t,
            stride_k_t=stride_k_t,
            stride_v_t=stride_v_t,
            stride_a_t=stride_a_t,
            stride_b_t=stride_b_t,
            stride_o_t=stride_o_t,
            stride_state_slot=stride_state_slot,
            stride_d_slot=stride_d_slot,
            stride_k_slot=stride_k_slot,
            stride_g_slot=stride_g_slot,
            stride_rawv_slot=stride_rawv_slot,
            stride_rawk_slot=stride_rawk_slot,
            stride_beta_slot=stride_beta_slot,
            stride_qsl=stride_qsl,
            stride_indices=stride_indices,
            stride_replay_indices=stride_replay_indices,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            BS=BS,
            BC=BC,
            NK=NK,
            BKT=BKT,
            MAX_CACHE_LEN=MAX_CACHE_LEN,
            SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
            STATE_FP32=STATE_FP32,
            STORE_EXACT_RAW=STORE_EXACT_RAW,
            STORE_D_RESIDUAL=STORE_D_RESIDUAL,
            STORE_K_RESIDUAL=STORE_K_RESIDUAL,
            IS_FLUSH=IS_FLUSH,
            NULL_BLOCK_ID=NULL_BLOCK_ID,
            DOT_PRECISION=DOT_PRECISION,
            DIRECT_SOLVE=DIRECT_SOLVE,
            SPEC_TOKENS=SPEC_TOKENS,
            EMPTY_HISTORY=False,
        )
