"""Fused TurboQuant decode attention kernel.

Reads packed uint8 KV cache directly during attention computation,
eliminating the need for a separate dequant buffer.

Supports asymmetric K/V bit widths (e.g., K=4bit V=2bit):
  K side: Q is split by K_VALS_PER_BYTE, codebook lookup uses K centroids
  V side: accumulators organized by V_VALS_PER_BYTE, codebook lookup uses V centroids

Uses N-way Split Dot Product (N = values_per_byte):
  4-bit (N=2): qk = dot(q_even, k_lo) + dot(q_odd, k_hi)
  2-bit (N=4): qk = dot(q_0, k_0) + dot(q_1, k_1) + dot(q_2, k_2) + dot(q_3, k_3)

Supports GQA (kv_group_num > 1) and MHA (kv_group_num == 1).
Supports 2-bit (4 values/byte) and 4-bit (2 values/byte).
"""

import triton
import triton.language as tl

_MIN_BLOCK_KV = 32


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _lookup_2bit(idx, c0, c1, c2, c3):
    """2-bit codebook lookup via predicated select (no scatter gather)."""
    return tl.where(idx == 0, c0, tl.where(idx == 1, c1, tl.where(idx == 2, c2, c3)))


@triton.jit
def _lookup_4bit_codebook(idx, c0, c1, c2, c3, c4, c5, c6, c7,
                          c8, c9, c10, c11, c12, c13, c14, c15):
    """4-bit codebook lookup via binary select tree (4 levels, 15 tl.where)."""
    lo = tl.where(
        (idx & 4) != 0,
        tl.where((idx & 2) != 0,
                 tl.where((idx & 1) != 0, c7, c6),
                 tl.where((idx & 1) != 0, c5, c4)),
        tl.where((idx & 2) != 0,
                 tl.where((idx & 1) != 0, c3, c2),
                 tl.where((idx & 1) != 0, c1, c0)),
    )
    hi = tl.where(
        (idx & 4) != 0,
        tl.where((idx & 2) != 0,
                 tl.where((idx & 1) != 0, c15, c14),
                 tl.where((idx & 1) != 0, c13, c12)),
        tl.where((idx & 2) != 0,
                 tl.where((idx & 1) != 0, c11, c10),
                 tl.where((idx & 1) != 0, c9, c8)),
    )
    return tl.where((idx & 8) != 0, hi, lo)


@triton.jit
def _lookup_4bit_uniform(idx, c0, c1, c2, c3, c4, c5, c6, c7,
                         c8, c9, c10, c11, c12, c13, c14, c15):
    """4-bit uniform dequant: 1 FMA replaces 15 tl.where selects."""
    step = (c15 - c0) * 0.06666666666666667  # 1/15
    return idx.to(tl.float32) * step + c0


@triton.jit
def _lookup_4bit(idx, c0, c1, c2, c3, c4, c5, c6, c7,
                 c8, c9, c10, c11, c12, c13, c14, c15,
                 UNIFORM: tl.constexpr = False):
    """4-bit lookup dispatcher: codebook or uniform based on constexpr flag."""
    if UNIFORM:
        return _lookup_4bit_uniform(idx, c0, c1, c2, c3, c4, c5, c6, c7,
                                    c8, c9, c10, c11, c12, c13, c14, c15)
    else:
        return _lookup_4bit_codebook(idx, c0, c1, c2, c3, c4, c5, c6, c7,
                                     c8, c9, c10, c11, c12, c13, c14, c15)


@triton.jit
def _fwd_tq_grouped_kernel_stage1(
    Q,
    K_Packed,
    V_Packed,
    K_DScale,
    V_DScale,
    K_Centroids,
    V_Centroids,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_kp_bs,
    stride_kp_h,
    stride_vp_bs,
    stride_vp_h,
    stride_kds_bs,
    stride_vds_bs,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    K_BLOCK_PACKED_DIM: tl.constexpr,
    V_BLOCK_PACKED_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    K_Lk_packed: tl.constexpr,
    V_Lv_packed: tl.constexpr,
    K_VALS_PER_BYTE: tl.constexpr,  # 2 for K=4-bit, 4 for K=2-bit
    K_BITS_PER_VAL: tl.constexpr,   # 4 for K=4-bit, 2 for K=2-bit
    K_BIT_MASK: tl.constexpr,       # 0x0F for K=4-bit, 0x03 for K=2-bit
    V_VALS_PER_BYTE: tl.constexpr,  # 2 for V=4-bit, 4 for V=2-bit
    V_BITS_PER_VAL: tl.constexpr,   # 4 for V=4-bit, 2 for V=2-bit
    V_BIT_MASK: tl.constexpr,       # 0x0F for V=4-bit, 0x03 for V=2-bit
    xai_temperature_len: tl.constexpr,
    UNIFORM: tl.constexpr = False,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    # Separate offset ranges for K and V packed dims
    offs_kp = tl.arange(0, K_BLOCK_PACKED_DIM)
    mask_kp = offs_kp < K_Lk_packed
    offs_vp = tl.arange(0, V_BLOCK_PACKED_DIM)
    mask_vp = offs_vp < V_Lv_packed

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)

    # V-side accumulators (organized by V_VALS_PER_BYTE, compiler eliminates unused)
    acc_0 = tl.zeros([BLOCK_H, V_BLOCK_PACKED_DIM], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, V_BLOCK_PACKED_DIM], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, V_BLOCK_PACKED_DIM], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, V_BLOCK_PACKED_DIM], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        # Preload K centroid values
        if K_VALS_PER_BYTE == 4:
            # K=2-bit: 4 centroids
            kc0 = tl.load(K_Centroids)
            kc1 = tl.load(K_Centroids + 1)
            kc2 = tl.load(K_Centroids + 2)
            kc3 = tl.load(K_Centroids + 3)
        else:
            # K=4-bit: 16 centroids
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

        # Preload V centroid values
        if V_VALS_PER_BYTE == 4:
            # V=2-bit: 4 centroids
            vc0 = tl.load(V_Centroids)
            vc1 = tl.load(V_Centroids + 1)
            vc2 = tl.load(V_Centroids + 2)
            vc3 = tl.load(V_Centroids + 3)
        else:
            # V=4-bit: 16 centroids
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

        # Load Q splits (strided by K_VALS_PER_BYTE, Q only used for QK dot)
        offs_q_0 = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + (K_VALS_PER_BYTE * offs_kp[None, :])
        offs_q_1 = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + (K_VALS_PER_BYTE * offs_kp[None, :] + 1)
        mask_q_0 = (mask_h[:, None]) & ((K_VALS_PER_BYTE * offs_kp[None, :]) < Lk)
        mask_q_1 = (mask_h[:, None]) & ((K_VALS_PER_BYTE * offs_kp[None, :] + 1) < Lk)

        q_0 = tl.load(Q + offs_q_0, mask=mask_q_0, other=0.0)
        q_1 = tl.load(Q + offs_q_1, mask=mask_q_1, other=0.0)

        if K_VALS_PER_BYTE == 4:
            offs_q_2 = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + (4 * offs_kp[None, :] + 2)
            offs_q_3 = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + (4 * offs_kp[None, :] + 3)
            mask_q_2 = (mask_h[:, None]) & ((4 * offs_kp[None, :] + 2) < Lk)
            mask_q_3 = (mask_h[:, None]) & ((4 * offs_kp[None, :] + 3) < Lk)
            q_2 = tl.load(Q + offs_q_2, mask=mask_q_2, other=0.0)
            q_3 = tl.load(Q + offs_q_3, mask=mask_q_3, other=0.0)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # --- K: load packed, sequential extract+lookup+dot (K params) ---
            offs_buf_kp = (
                kv_loc[None, :] * stride_kp_bs
                + cur_kv_head * stride_kp_h
                + offs_kp[:, None]
            )
            packed_k = tl.load(
                K_Packed + offs_buf_kp,
                mask=(offs_n[None, :] < split_kv_end) & (mask_kp[:, None]),
                other=0,
            )

            k_idx_0 = (packed_k & K_BIT_MASK).to(tl.int32)
            if K_VALS_PER_BYTE == 4:
                k_0 = _lookup_2bit(k_idx_0, kc0, kc1, kc2, kc3)
            else:
                k_0 = _lookup_4bit(k_idx_0, kc0, kc1, kc2, kc3, kc4, kc5, kc6, kc7,
                                   kc8, kc9, kc10, kc11, kc12, kc13, kc14, kc15, UNIFORM=UNIFORM)
            qk = tl.dot(q_0, k_0.to(q_0.dtype))

            k_idx_1 = ((packed_k >> K_BITS_PER_VAL) & K_BIT_MASK).to(tl.int32)
            if K_VALS_PER_BYTE == 4:
                k_1 = _lookup_2bit(k_idx_1, kc0, kc1, kc2, kc3)
            else:
                k_1 = _lookup_4bit(k_idx_1, kc0, kc1, kc2, kc3, kc4, kc5, kc6, kc7,
                                   kc8, kc9, kc10, kc11, kc12, kc13, kc14, kc15, UNIFORM=UNIFORM)
            qk += tl.dot(q_1, k_1.to(q_1.dtype))

            if K_VALS_PER_BYTE == 4:
                k_idx_2 = ((packed_k >> (2 * K_BITS_PER_VAL)) & K_BIT_MASK).to(tl.int32)
                k_2 = _lookup_2bit(k_idx_2, kc0, kc1, kc2, kc3)
                qk += tl.dot(q_2, k_2.to(q_2.dtype))

                k_idx_3 = ((packed_k >> (3 * K_BITS_PER_VAL)) & K_BIT_MASK).to(tl.int32)
                k_3 = _lookup_2bit(k_idx_3, kc0, kc1, kc2, kc3)
                qk += tl.dot(q_3, k_3.to(q_3.dtype))

            # K dequant scale
            k_dscale = tl.load(
                K_DScale + kv_loc * stride_kds_bs + cur_kv_head,
                mask=offs_n < split_kv_end, other=1.0,
            ).to(tl.float32)
            qk *= k_dscale[None, :]
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # --- Online softmax (V not loaded yet → fewer live registers) ---
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc_0 *= re_scale[:, None]
            acc_1 *= re_scale[:, None]
            if V_VALS_PER_BYTE == 4:
                acc_2 *= re_scale[:, None]
                acc_3 *= re_scale[:, None]

            # V dequant scale
            v_dscale = tl.load(
                V_DScale + kv_loc * stride_vds_bs + cur_kv_head,
                mask=offs_n < split_kv_end, other=1.0,
            ).to(tl.float32)
            p_scaled = p * v_dscale[None, :]

            # --- V: load packed AFTER softmax (V params) ---
            offs_buf_vp = (
                kv_loc[:, None] * stride_vp_bs
                + cur_kv_head * stride_vp_h
                + offs_vp[None, :]
            )
            packed_v = tl.load(
                V_Packed + offs_buf_vp,
                mask=(offs_n[:, None] < split_kv_end) & (mask_vp[None, :]),
                other=0,
            )

            v_idx_0 = (packed_v & V_BIT_MASK).to(tl.int32)
            if V_VALS_PER_BYTE == 4:
                v_0 = _lookup_2bit(v_idx_0, vc0, vc1, vc2, vc3)
            else:
                v_0 = _lookup_4bit(v_idx_0, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7,
                                   vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, UNIFORM=UNIFORM)
            acc_0 += tl.dot(p_scaled.to(v_0.dtype), v_0)

            v_idx_1 = ((packed_v >> V_BITS_PER_VAL) & V_BIT_MASK).to(tl.int32)
            if V_VALS_PER_BYTE == 4:
                v_1 = _lookup_2bit(v_idx_1, vc0, vc1, vc2, vc3)
            else:
                v_1 = _lookup_4bit(v_idx_1, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7,
                                   vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15, UNIFORM=UNIFORM)
            acc_1 += tl.dot(p_scaled.to(v_1.dtype), v_1)

            if V_VALS_PER_BYTE == 4:
                v_idx_2 = ((packed_v >> (2 * V_BITS_PER_VAL)) & V_BIT_MASK).to(tl.int32)
                v_2 = _lookup_2bit(v_idx_2, vc0, vc1, vc2, vc3)
                acc_2 += tl.dot(p_scaled.to(v_2.dtype), v_2)

                v_idx_3 = ((packed_v >> (3 * V_BITS_PER_VAL)) & V_BIT_MASK).to(tl.int32)
                v_3 = _lookup_2bit(v_idx_3, vc0, vc1, vc2, vc3)
                acc_3 += tl.dot(p_scaled.to(v_3.dtype), v_3)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # Store: V-side N-way interleaved output
        base_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
        )

        offs_out_0 = V_VALS_PER_BYTE * tl.arange(0, V_BLOCK_PACKED_DIM)
        offs_out_1 = V_VALS_PER_BYTE * tl.arange(0, V_BLOCK_PACKED_DIM) + 1
        mask_out_0 = offs_out_0 < Lv
        mask_out_1 = offs_out_1 < Lv

        tl.store(Att_Out + base_mid_o + offs_out_0[None, :],
                 acc_0 / e_sum[:, None], mask=(mask_h[:, None]) & (mask_out_0[None, :]))
        tl.store(Att_Out + base_mid_o + offs_out_1[None, :],
                 acc_1 / e_sum[:, None], mask=(mask_h[:, None]) & (mask_out_1[None, :]))

        if V_VALS_PER_BYTE == 4:
            offs_out_2 = 4 * tl.arange(0, V_BLOCK_PACKED_DIM) + 2
            offs_out_3 = 4 * tl.arange(0, V_BLOCK_PACKED_DIM) + 3
            mask_out_2 = offs_out_2 < Lv
            mask_out_3 = offs_out_3 < Lv
            tl.store(Att_Out + base_mid_o + offs_out_2[None, :],
                     acc_2 / e_sum[:, None], mask=(mask_h[:, None]) & (mask_out_2[None, :]))
            tl.store(Att_Out + base_mid_o + offs_out_3[None, :],
                     acc_3 / e_sum[:, None], mask=(mask_h[:, None]) & (mask_out_3[None, :]))

        # Store LSE
        offs_mid_lse = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(Att_Lse + offs_mid_lse, e_max + tl.log(e_sum), mask=mask_h)


def _bit_params(bit_width):
    """Return (VALS_PER_BYTE, BITS_PER_VAL, BIT_MASK) for a given bit width."""
    if bit_width == 2:
        return 4, 2, 0x03
    elif bit_width == 4:
        return 2, 4, 0x0F
    else:
        raise ValueError(f"Unsupported bit_width: {bit_width}")


def _tq_decode_grouped_att_m_fwd(
    q, k_packed, v_packed, k_dscale, v_dscale,
    k_centroids, v_centroids, o, att_out, att_lse, kv_indptr, kv_indices,
    num_kv_splits, max_kv_splits, sm_scale, logit_cap,
    k_bit_width, v_bit_width,
    xai_temperature_len=-1,
    uniform=False,
):
    Lk = q.shape[-1]
    Lv = Lk
    assert o.shape[-1] == Lk, (
        f"TurboQuant: v_head_dim ({o.shape[-1]}) != qk_head_dim ({Lk})"
    )
    K_Lk_packed = k_packed.shape[-1]
    V_Lv_packed = v_packed.shape[-1]

    K_VALS_PER_BYTE, K_BITS_PER_VAL, K_BIT_MASK = _bit_params(k_bit_width)
    V_VALS_PER_BYTE, V_BITS_PER_VAL, V_BIT_MASK = _bit_params(v_bit_width)

    K_BLOCK_PACKED_DIM = triton.next_power_of_2(K_Lk_packed)
    V_BLOCK_PACKED_DIM = triton.next_power_of_2(V_Lv_packed)
    BLOCK_N = 16
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_packed.shape[1]
    BLOCK_H = min(16, kv_group_num)

    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        max_kv_splits,
    )

    _fwd_tq_grouped_kernel_stage1[grid](
        q, k_packed, v_packed, k_dscale, v_dscale,
        k_centroids, v_centroids,
        sm_scale, kv_indptr, kv_indices, att_out, att_lse,
        num_kv_splits,
        q.stride(0), q.stride(1),
        k_packed.stride(0), k_packed.stride(1),
        v_packed.stride(0), v_packed.stride(1),
        k_dscale.stride(0), v_dscale.stride(0),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        K_BLOCK_PACKED_DIM=K_BLOCK_PACKED_DIM,
        V_BLOCK_PACKED_DIM=V_BLOCK_PACKED_DIM,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
        K_Lk_packed=K_Lk_packed,
        V_Lv_packed=V_Lv_packed,
        K_VALS_PER_BYTE=K_VALS_PER_BYTE,
        K_BITS_PER_VAL=K_BITS_PER_VAL,
        K_BIT_MASK=K_BIT_MASK,
        V_VALS_PER_BYTE=V_VALS_PER_BYTE,
        V_BITS_PER_VAL=V_BITS_PER_VAL,
        V_BIT_MASK=V_BIT_MASK,
        xai_temperature_len=xai_temperature_len,
        UNIFORM=uniform,
    )


def tq_decode_attention_fwd(
    q, k_packed, v_packed, k_dscale, v_dscale,
    k_centroids, v_centroids, o, kv_indptr, kv_indices, attn_logits, attn_lse,
    num_kv_splits, max_kv_splits, sm_scale,
    k_bit_width=4, v_bit_width=4,
    logit_cap=0.0, sinks=None, xai_temperature_len=-1,
    uniform=False,
):
    """Fused TurboQuant decode attention: reads packed uint8 KV directly.

    Supports asymmetric K/V bit widths (e.g., K=4bit V=2bit).
    Supports 2-bit (4-way split) and 4-bit (2-way split).
    """
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        _decode_softmax_reducev_fwd,
    )

    assert max_kv_splits == attn_logits.shape[2]
    assert k_bit_width in (2, 4), f"Unsupported K bit_width: {k_bit_width}"
    assert v_bit_width in (2, 4), f"Unsupported V bit_width: {v_bit_width}"

    # Stage 1: fused TQ attention with packed KV
    _tq_decode_grouped_att_m_fwd(
        q, k_packed, v_packed, k_dscale, v_dscale,
        k_centroids, v_centroids, o, attn_logits, attn_lse, kv_indptr, kv_indices,
        num_kv_splits, max_kv_splits, sm_scale, logit_cap,
        k_bit_width, v_bit_width,
        xai_temperature_len=xai_temperature_len,
        uniform=uniform,
    )

    # Stage 2: reuse standard softmax reduce
    _decode_softmax_reducev_fwd(
        attn_logits, attn_lse, q, o,
        1.0,  # v_scale = 1.0 (dequant scale applied in stage1)
        o,    # v_buffer: only .shape[-1] used for Lv
        kv_indptr, num_kv_splits, max_kv_splits, sinks,
    )
