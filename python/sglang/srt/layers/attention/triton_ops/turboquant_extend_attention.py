"""Fused TurboQuant extend attention kernel.

Reads packed uint8 KV cache directly during extend (prefill) attention,
eliminating the expensive eager dequant (`batched_dequantize_rotspace`).

Supports asymmetric K/V bit widths (e.g., K=4bit V=2bit):
  K side: Q is split by K_VALS_PER_BYTE, K uses K codebook
  V side: accumulators organized by V_VALS_PER_BYTE, V uses V codebook

Both the prefix stage (packed uint8 from KV pool) and extend stage (fresh bf16)
use N-way Split Dot Product throughout, with a single interleaved output store.
"""

import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.turboquant_decode_attention import (
    _bit_params,
    _lookup_2bit,
    _lookup_4bit,
    tanh,
)


@triton.jit
def _fwd_tq_extend_kernel(
    # Extend tensors (contiguous bf16, already in rotspace)
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    # Packed KV pool buffers (uint8)
    K_Packed,
    V_Packed,
    # Dequant scales (bf16, per token-head)
    K_DScale,
    V_DScale,
    # Codebook centroids (float32), separate for K and V
    K_Centroids,
    V_Centroids,
    # Standard extend attention indirection
    qo_indptr,
    kv_indptr,
    kv_indices,
    # Mask
    mask_ptr,
    mask_indptr,
    # Sink
    sink_ptr,
    # Scalars
    sm_scale,
    kv_group_num,
    # Q/K/V extend strides
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    # Packed buffer strides
    stride_kp_bs,
    stride_kp_h,
    stride_vp_bs,
    stride_vp_h,
    # Dequant scale strides
    stride_kds_bs,
    stride_vds_bs,
    # Constexpr — K side
    K_BLOCK_PACKED_DIM: tl.constexpr,
    K_Lk_packed: tl.constexpr,
    K_VALS_PER_BYTE: tl.constexpr,
    K_BITS_PER_VAL: tl.constexpr,
    K_BIT_MASK: tl.constexpr,
    # Constexpr — V side
    V_BLOCK_PACKED_DIM: tl.constexpr,
    V_Lv_packed: tl.constexpr,
    V_VALS_PER_BYTE: tl.constexpr,
    V_BITS_PER_VAL: tl.constexpr,
    V_BIT_MASK: tl.constexpr,
    # Constexpr — common
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_SINK: tl.constexpr,
    UNIFORM: tl.constexpr = False,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    # Separate offset ranges for K and V packed dims
    offs_kp = tl.arange(0, K_BLOCK_PACKED_DIM)
    mask_kp = offs_kp < K_Lk_packed
    offs_vp = tl.arange(0, V_BLOCK_PACKED_DIM)
    mask_vp = offs_vp < V_Lv_packed
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    if xai_temperature_len > 0:
        offs_qidx = cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        xai_temperature_reg = tl.where(
            offs_qidx > xai_temperature_len,
            tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale,
            1.0,
        )

    # --- Preload K centroids ---
    if K_VALS_PER_BYTE == 4:
        kc0 = tl.load(K_Centroids)
        kc1 = tl.load(K_Centroids + 1)
        kc2 = tl.load(K_Centroids + 2)
        kc3 = tl.load(K_Centroids + 3)
    else:
        kc0 = tl.load(K_Centroids)
        kc1 = tl.load(K_Centroids + 1)
        kc2 = tl.load(K_Centroids + 2)
        kc3 = tl.load(K_Centroids + 3)
        kc4 = tl.load(K_Centroids + 4)
        kc5 = tl.load(K_Centroids + 5)
        kc6 = tl.load(K_Centroids + 6)
        kc7 = tl.load(K_Centroids + 7)
        kc8 = tl.load(K_Centroids + 8)
        kc9 = tl.load(K_Centroids + 9)
        kc10 = tl.load(K_Centroids + 10)
        kc11 = tl.load(K_Centroids + 11)
        kc12 = tl.load(K_Centroids + 12)
        kc13 = tl.load(K_Centroids + 13)
        kc14 = tl.load(K_Centroids + 14)
        kc15 = tl.load(K_Centroids + 15)

    # --- Preload V centroids ---
    if V_VALS_PER_BYTE == 4:
        vc0 = tl.load(V_Centroids)
        vc1 = tl.load(V_Centroids + 1)
        vc2 = tl.load(V_Centroids + 2)
        vc3 = tl.load(V_Centroids + 3)
    else:
        vc0 = tl.load(V_Centroids)
        vc1 = tl.load(V_Centroids + 1)
        vc2 = tl.load(V_Centroids + 2)
        vc3 = tl.load(V_Centroids + 3)
        vc4 = tl.load(V_Centroids + 4)
        vc5 = tl.load(V_Centroids + 5)
        vc6 = tl.load(V_Centroids + 6)
        vc7 = tl.load(V_Centroids + 7)
        vc8 = tl.load(V_Centroids + 8)
        vc9 = tl.load(V_Centroids + 9)
        vc10 = tl.load(V_Centroids + 10)
        vc11 = tl.load(V_Centroids + 11)
        vc12 = tl.load(V_Centroids + 12)
        vc13 = tl.load(V_Centroids + 13)
        vc14 = tl.load(V_Centroids + 14)
        vc15 = tl.load(V_Centroids + 15)

    # --- Load split Q vectors (strided by K_VALS_PER_BYTE, Q only for QK dot) ---
    q_base = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
    )
    offs_q_0 = q_base + (K_VALS_PER_BYTE * offs_kp[None, :])
    offs_q_1 = q_base + (K_VALS_PER_BYTE * offs_kp[None, :] + 1)
    mask_q_0 = mask_m[:, None] & ((K_VALS_PER_BYTE * offs_kp[None, :]) < Lq)
    mask_q_1 = mask_m[:, None] & ((K_VALS_PER_BYTE * offs_kp[None, :] + 1) < Lq)

    q_0 = tl.load(Q_Extend + offs_q_0, mask=mask_q_0, other=0.0)
    q_1 = tl.load(Q_Extend + offs_q_1, mask=mask_q_1, other=0.0)

    if K_VALS_PER_BYTE == 4:
        offs_q_2 = q_base + (4 * offs_kp[None, :] + 2)
        offs_q_3 = q_base + (4 * offs_kp[None, :] + 3)
        mask_q_2 = mask_m[:, None] & ((4 * offs_kp[None, :] + 2) < Lq)
        mask_q_3 = mask_m[:, None] & ((4 * offs_kp[None, :] + 3) < Lq)
        q_2 = tl.load(Q_Extend + offs_q_2, mask=mask_q_2, other=0.0)
        q_3 = tl.load(Q_Extend + offs_q_3, mask=mask_q_3, other=0.0)

    # --- Initialize V-side accumulators ---
    acc_0 = tl.zeros([BLOCK_M, V_BLOCK_PACKED_DIM], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_M, V_BLOCK_PACKED_DIM], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_M, V_BLOCK_PACKED_DIM], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_M, V_BLOCK_PACKED_DIM], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # =========================================================
    # STAGE 1: Prefix loop — packed uint8 from KV pool
    # =========================================================
    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        final_mask = mask_m[:, None] & mask_n[None, :]

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * cur_seq_len
                + start_n
                + offs_n[None, :],
                mask=final_mask,
                other=0,
            )
            final_mask &= custom_mask

        # Load kv_loc via indirection
        offs_kv_loc = tl.load(
            kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
            mask=mask_n,
            other=0,
        )

        # --- K: load packed uint8, K codebook lookup (K params) ---
        offs_buf_kp = (
            offs_kv_loc[None, :] * stride_kp_bs
            + cur_kv_head * stride_kp_h
            + offs_kp[:, None]
        )
        packed_k = tl.load(
            K_Packed + offs_buf_kp,
            mask=mask_n[None, :] & mask_kp[:, None],
            other=0,
        )

        k_idx_0 = (packed_k & K_BIT_MASK).to(tl.int32)
        k_idx_1 = ((packed_k >> K_BITS_PER_VAL) & K_BIT_MASK).to(tl.int32)
        if K_VALS_PER_BYTE == 4:
            k_0 = _lookup_2bit(k_idx_0, kc0, kc1, kc2, kc3)
            k_1 = _lookup_2bit(k_idx_1, kc0, kc1, kc2, kc3)
        else:
            k_0 = _lookup_4bit(k_idx_0, kc0, kc1, kc2, kc3, kc4, kc5, kc6, kc7,
                               kc8, kc9, kc10, kc11, kc12, kc13, kc14, kc15, UNIFORM=UNIFORM)
            k_1 = _lookup_4bit(k_idx_1, kc0, kc1, kc2, kc3, kc4, kc5, kc6, kc7,
                               kc8, kc9, kc10, kc11, kc12, kc13, kc14, kc15, UNIFORM=UNIFORM)

        qk = tl.dot(q_0, k_0.to(q_0.dtype)) + tl.dot(q_1, k_1.to(q_1.dtype))

        if K_VALS_PER_BYTE == 4:
            k_idx_2 = ((packed_k >> (2 * K_BITS_PER_VAL)) & K_BIT_MASK).to(tl.int32)
            k_idx_3 = ((packed_k >> (3 * K_BITS_PER_VAL)) & K_BIT_MASK).to(tl.int32)
            k_2 = _lookup_2bit(k_idx_2, kc0, kc1, kc2, kc3)
            k_3 = _lookup_2bit(k_idx_3, kc0, kc1, kc2, kc3)
            qk += tl.dot(q_2, k_2.to(q_2.dtype)) + tl.dot(q_3, k_3.to(q_3.dtype))

        # K dequant scale
        k_dscale = tl.load(
            K_DScale + offs_kv_loc * stride_kds_bs + cur_kv_head,
            mask=mask_n, other=1.0,
        ).to(tl.float32)
        qk *= k_dscale[None, :]
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if xai_temperature_len > 0:
            qk *= xai_temperature_reg[:, None]

        qk = tl.where(final_mask, qk, float("-inf"))

        # --- V: load packed uint8, V codebook lookup (V params) ---
        offs_buf_vp = (
            offs_kv_loc[:, None] * stride_vp_bs
            + cur_kv_head * stride_vp_h
            + offs_vp[None, :]
        )
        packed_v = tl.load(
            V_Packed + offs_buf_vp,
            mask=mask_n[:, None] & mask_vp[None, :],
            other=0,
        )
        v_idx_0 = (packed_v & V_BIT_MASK).to(tl.int32)
        v_idx_1 = ((packed_v >> V_BITS_PER_VAL) & V_BIT_MASK).to(tl.int32)
        if V_VALS_PER_BYTE == 4:
            v_0 = _lookup_2bit(v_idx_0, vc0, vc1, vc2, vc3)
            v_1 = _lookup_2bit(v_idx_1, vc0, vc1, vc2, vc3)
        else:
            v_0 = _lookup_4bit(v_idx_0, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7,
                               vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, UNIFORM=UNIFORM)
            v_1 = _lookup_4bit(v_idx_1, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7,
                               vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, UNIFORM=UNIFORM)
        if V_VALS_PER_BYTE == 4:
            v_idx_2 = ((packed_v >> (2 * V_BITS_PER_VAL)) & V_BIT_MASK).to(tl.int32)
            v_idx_3 = ((packed_v >> (3 * V_BITS_PER_VAL)) & V_BIT_MASK).to(tl.int32)
            v_2 = _lookup_2bit(v_idx_2, vc0, vc1, vc2, vc3)
            v_3 = _lookup_2bit(v_idx_3, vc0, vc1, vc2, vc3)

        # Online softmax
        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        n_e_max = tl.maximum(row_max_fixed, e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        acc_0 *= re_scale[:, None]
        acc_1 *= re_scale[:, None]
        if V_VALS_PER_BYTE == 4:
            acc_2 *= re_scale[:, None]
            acc_3 *= re_scale[:, None]

        # V dequant scale + accumulate
        v_dscale = tl.load(
            V_DScale + offs_kv_loc * stride_vds_bs + cur_kv_head,
            mask=mask_n, other=1.0,
        ).to(tl.float32)
        p_scaled = p * v_dscale[None, :]

        acc_0 += tl.dot(p_scaled.to(v_0.dtype), v_0)
        acc_1 += tl.dot(p_scaled.to(v_1.dtype), v_1)
        if V_VALS_PER_BYTE == 4:
            acc_2 += tl.dot(p_scaled.to(v_2.dtype), v_2)
            acc_3 += tl.dot(p_scaled.to(v_3.dtype), v_3)

        e_max = n_e_max

    # =========================================================
    # STAGE 2: Extend/triangle loop — bf16 with strided loads
    # K uses K_VALS_PER_BYTE stride, V uses V_VALS_PER_BYTE stride
    # =========================================================
    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )

    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * cur_seq_len
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=final_mask,
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            final_mask &= custom_mask
        elif IS_CAUSAL:
            mask_causal = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causal &= mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_causal
        else:
            final_mask &= mask_m[:, None] & mask_n[None, :]

        # Load K_Extend with K stride pattern (transposed: [K_PACKED_DIM, BLOCK_N])
        k_ext_base = (
            (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
        )
        offs_k_0 = k_ext_base + (K_VALS_PER_BYTE * offs_kp[:, None])
        offs_k_1 = k_ext_base + (K_VALS_PER_BYTE * offs_kp[:, None] + 1)
        mask_k = mask_n[None, :] & mask_kp[:, None]

        k_ext_0 = tl.load(K_Extend + offs_k_0, mask=mask_k, other=0.0)
        k_ext_1 = tl.load(K_Extend + offs_k_1, mask=mask_k, other=0.0)

        qk = tl.dot(q_0, k_ext_0.to(q_0.dtype)) + tl.dot(q_1, k_ext_1.to(q_1.dtype))

        if K_VALS_PER_BYTE == 4:
            offs_k_2 = k_ext_base + (4 * offs_kp[:, None] + 2)
            offs_k_3 = k_ext_base + (4 * offs_kp[:, None] + 3)
            k_ext_2 = tl.load(K_Extend + offs_k_2, mask=mask_k, other=0.0)
            k_ext_3 = tl.load(K_Extend + offs_k_3, mask=mask_k, other=0.0)
            qk += tl.dot(q_2, k_ext_2.to(q_2.dtype)) + tl.dot(q_3, k_ext_3.to(q_3.dtype))

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if xai_temperature_len > 0:
            qk *= xai_temperature_reg[:, None]

        qk = tl.where(final_mask, qk, float("-inf"))

        # Online softmax
        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        n_e_max = tl.maximum(row_max_fixed, e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        acc_0 *= re_scale[:, None]
        acc_1 *= re_scale[:, None]
        if V_VALS_PER_BYTE == 4:
            acc_2 *= re_scale[:, None]
            acc_3 *= re_scale[:, None]

        # Load V_Extend with V stride pattern [BLOCK_N, V_PACKED_DIM]
        v_ext_base = (
            (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
        )
        offs_v_0 = v_ext_base + (V_VALS_PER_BYTE * offs_vp[None, :])
        offs_v_1 = v_ext_base + (V_VALS_PER_BYTE * offs_vp[None, :] + 1)
        mask_v = mask_n[:, None] & mask_vp[None, :]

        v_ext_0 = tl.load(V_Extend + offs_v_0, mask=mask_v, other=0.0)
        v_ext_1 = tl.load(V_Extend + offs_v_1, mask=mask_v, other=0.0)

        p_cast = p.to(v_ext_0.dtype)
        acc_0 += tl.dot(p_cast, v_ext_0)
        acc_1 += tl.dot(p_cast, v_ext_1)

        if V_VALS_PER_BYTE == 4:
            offs_v_2 = v_ext_base + (4 * offs_vp[None, :] + 2)
            offs_v_3 = v_ext_base + (4 * offs_vp[None, :] + 3)
            v_ext_2 = tl.load(V_Extend + offs_v_2, mask=mask_v, other=0.0)
            v_ext_3 = tl.load(V_Extend + offs_v_3, mask=mask_v, other=0.0)
            acc_2 += tl.dot(p_cast, v_ext_2)
            acc_3 += tl.dot(p_cast, v_ext_3)

        e_max = n_e_max

    # =========================================================
    # Sink handling
    # =========================================================
    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    # =========================================================
    # Output store — V-side interleaved N-way accumulators
    # =========================================================
    o_base = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
    )

    offs_out_0 = V_VALS_PER_BYTE * tl.arange(0, V_BLOCK_PACKED_DIM)
    offs_out_1 = V_VALS_PER_BYTE * tl.arange(0, V_BLOCK_PACKED_DIM) + 1
    mask_out_0 = (offs_out_0 < Lv)
    mask_out_1 = (offs_out_1 < Lv)

    tl.store(
        O_Extend + o_base + offs_out_0[None, :],
        acc_0 / deno[:, None],
        mask=mask_m[:, None] & mask_out_0[None, :],
    )
    tl.store(
        O_Extend + o_base + offs_out_1[None, :],
        acc_1 / deno[:, None],
        mask=mask_m[:, None] & mask_out_1[None, :],
    )

    if V_VALS_PER_BYTE == 4:
        offs_out_2 = 4 * tl.arange(0, V_BLOCK_PACKED_DIM) + 2
        offs_out_3 = 4 * tl.arange(0, V_BLOCK_PACKED_DIM) + 3
        mask_out_2 = (offs_out_2 < Lv)
        mask_out_3 = (offs_out_3 < Lv)
        tl.store(
            O_Extend + o_base + offs_out_2[None, :],
            acc_2 / deno[:, None],
            mask=mask_m[:, None] & mask_out_2[None, :],
        )
        tl.store(
            O_Extend + o_base + offs_out_3[None, :],
            acc_3 / deno[:, None],
            mask=mask_m[:, None] & mask_out_3[None, :],
        )


def tq_extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_packed,
    v_packed,
    k_dscale,
    v_dscale,
    k_centroids,
    v_centroids,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    sm_scale=None,
    k_bit_width=4,
    v_bit_width=4,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    uniform=False,
):
    """Fused TurboQuant extend attention: reads packed uint8 KV directly.

    Supports asymmetric K/V bit widths (e.g., K=4bit V=2bit).
    Both prefix (packed KV pool) and extend (fresh bf16) stages use
    N-way split dot products, with interleaved output store.
    """
    assert k_bit_width in (2, 4), f"Unsupported K bit_width: {k_bit_width}"
    assert v_bit_width in (2, 4), f"Unsupported V bit_width: {v_bit_width}"

    Lq = q_extend.shape[-1]
    Lv = Lq  # TQ constrains Lk == Lv
    K_Lk_packed = k_packed.shape[-1]
    V_Lv_packed = v_packed.shape[-1]

    K_VALS_PER_BYTE, K_BITS_PER_VAL, K_BIT_MASK = _bit_params(k_bit_width)
    V_VALS_PER_BYTE, V_BITS_PER_VAL, V_BIT_MASK = _bit_params(v_bit_width)

    K_BLOCK_PACKED_DIM = triton.next_power_of_2(K_Lk_packed)
    V_BLOCK_PACKED_DIM = triton.next_power_of_2(V_Lv_packed)
    # Smaller BLOCK_M than standard extend (128) due to N-way accumulator pressure
    BLOCK_M = 64
    BLOCK_N = 64

    sm_scale = sm_scale or 1.0 / (Lq ** 0.5)
    batch_size = qo_indptr.shape[0] - 1
    head_num = q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_packed.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None
    HAS_SINK = sinks is not None

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))

    _fwd_tq_extend_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_packed,
        v_packed,
        k_dscale,
        v_dscale,
        k_centroids,
        v_centroids,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sinks,
        sm_scale,
        kv_group_num,
        q_extend.stride(0), q_extend.stride(1),
        k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1),
        o_extend.stride(0), o_extend.stride(1),
        k_packed.stride(0), k_packed.stride(1),
        v_packed.stride(0), v_packed.stride(1),
        k_dscale.stride(0), v_dscale.stride(0),
        K_BLOCK_PACKED_DIM=K_BLOCK_PACKED_DIM,
        K_Lk_packed=K_Lk_packed,
        K_VALS_PER_BYTE=K_VALS_PER_BYTE,
        K_BITS_PER_VAL=K_BITS_PER_VAL,
        K_BIT_MASK=K_BIT_MASK,
        V_BLOCK_PACKED_DIM=V_BLOCK_PACKED_DIM,
        V_Lv_packed=V_Lv_packed,
        V_VALS_PER_BYTE=V_VALS_PER_BYTE,
        V_BITS_PER_VAL=V_BITS_PER_VAL,
        V_BIT_MASK=V_BIT_MASK,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        Lq=Lq,
        Lv=Lv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        IS_CAUSAL=is_causal,
        HAS_SINK=HAS_SINK,
        UNIFORM=uniform,
        num_warps=4,
        num_stages=2,
    )
