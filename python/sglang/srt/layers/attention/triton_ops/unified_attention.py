# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py
# Future work:
# - hardware specific tile size tuning
# - continuous access to extending k/v

import logging

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
if _is_cuda:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()
_is_hip = is_hip()


logger = logging.getLogger(__name__)
float8_dtype = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
float8_info = torch.finfo(float8_dtype)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    return x * (2 * tl.sigmoid(2 * S / x) - 1)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_cache_ptr,  # [num_tokens, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    kv_indptr_ptr,  # [num_seqs+1]
    kv_indices_ptr,  # [total_kv_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    softcap,  # float32
    mask_ptr,  # [\sum_{i<batch_size}(q_len_i * kv_len_i)]
    mask_indptr,  # [batch_size+1]
    window_start_pos_ptr,  # [batch_size], absolute key start for window kv
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_t: tl.int64,  # int
    stride_k_cache_h: tl.int64,  # int
    stride_k_cache_d: tl.constexpr,  # int
    stride_v_cache_t: tl.int64,  # int
    stride_v_cache_h: tl.int64,  # int
    stride_v_cache_d: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    USE_CUSTOM_MASK: tl.constexpr,  # bool
    xai_temperature_len: tl.constexpr,
    USE_WINDOW_START_POS: tl.constexpr,  # bool
    is_causal: tl.constexpr,  # bool
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + seq_idx)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    kv_start = tl.load(kv_indptr_ptr + seq_idx)
    kv_end = tl.load(kv_indptr_ptr + seq_idx + 1)
    seq_len = kv_end - kv_start

    if USE_WINDOW_START_POS:
        window_start = tl.load(window_start_pos_ptr + seq_idx)
    else:
        window_start = 0

    # context length for this particular sequence
    context_len = seq_len - cur_batch_query_len

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    if not is_causal:
        num_tiles = cdiv_fn(seq_len, TILE_SIZE)

    # ---- Sliding-window tile pruning -------------------- (for causal models only, in the future we could extend this optimization to non-causal models)
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0 and is_causal:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        kv_mask = seq_offset < seq_len

        physical_token_idx = tl.load(
            kv_indices_ptr + kv_start + seq_offset,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        v_offset = (
            physical_token_idx[:, None] * stride_v_cache_t
            + kv_head_idx * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
        )
        k_offset = (
            physical_token_idx[None, :] * stride_k_cache_t
            + kv_head_idx * stride_k_cache_h
            + offs_d[:, None] * stride_k_cache_d
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & kv_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * k_scale).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & kv_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * v_scale).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask: causal by default (key <= query)
        query_abs_pos = window_start + context_len + query_pos[:, None]
        key_abs_pos = window_start + seq_offset[None, :]
        seq_mask = key_abs_pos <= query_abs_pos

        if xai_temperature_len > 0:
            offs_qidx = query_abs_pos
            xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
            _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
            xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if xai_temperature_len > 0:
            S *= xai_temperature_reg

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        final_mask = query_mask_1[:, None] & query_mask_0[:, None]

        if SLIDING_WINDOW > 0:
            final_mask = final_mask & ((query_abs_pos - key_abs_pos) < SLIDING_WINDOW)

        if USE_CUSTOM_MASK:
            custom_mask = (
                tl.load(
                    mask_ptr
                    + cur_seq_mask_start_idx
                    + query_pos[:, None] * seq_len
                    + seq_offset,
                    mask=(query_mask_0[:, None] & kv_mask),
                    other=0,
                )
                != 0
            )
            final_mask &= custom_mask
        elif is_causal:
            final_mask &= seq_mask
        else:
            final_mask &= kv_mask[None, :]

        S = tl.where(
            final_mask,
            S,
            float("-inf"),
        )

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]
    if USE_FP8:
        acc = acc * out_scale
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size_padded]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_cache_ptr,  # [num_tokens, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    kv_indptr_ptr,  # [num_seqs+1]
    kv_indices_ptr,  # [total_kv_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    window_start_pos_ptr,  # [batch_size], absolute key start for window kv
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_t: tl.int64,  # int, token
    stride_k_cache_h: tl.int64,  # int, head
    stride_k_cache_d: tl.constexpr,  # int, dimension
    stride_v_cache_t: tl.int64,  # int
    stride_v_cache_h: tl.int64,  # int
    stride_v_cache_d: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    xai_temperature_len: tl.constexpr,
    USE_WINDOW_START_POS: tl.constexpr,  # bool
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    kv_start = tl.load(kv_indptr_ptr + seq_idx)
    kv_end = tl.load(kv_indptr_ptr + seq_idx + 1)
    seq_len = kv_end - kv_start

    if USE_WINDOW_START_POS:
        window_start = tl.load(window_start_pos_ptr + seq_idx)
    else:
        window_start = 0

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(
        max(segm_idx * tiles_per_segment, tile_start),
        min((segm_idx + 1) * tiles_per_segment, tile_end),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        kv_mask = seq_offset < seq_len

        physical_token_idx = tl.load(
            kv_indices_ptr + kv_start + seq_offset,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        v_offset = (
            physical_token_idx[:, None] * stride_v_cache_t
            + kv_head_idx * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
        )
        k_offset = (
            physical_token_idx[None, :] * stride_k_cache_t
            + kv_head_idx * stride_k_cache_h
            + offs_d[:, None] * stride_k_cache_d
        )

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & kv_mask[None, :],
            other=0.0,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * k_scale).to(Q.dtype)
        else:
            K = K_load

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & kv_mask[:, None],
            other=0.0,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * v_scale).to(Q.dtype)
        else:
            V = V_load

        # Compute attention mask
        query_abs_pos = window_start + context_len + query_pos[:, None]
        key_abs_pos = window_start + seq_offset[None, :]
        seq_mask = key_abs_pos <= query_abs_pos

        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((query_abs_pos - key_abs_pos) < SLIDING_WINDOW)

        if xai_temperature_len > 0:
            offs_qidx = query_abs_pos
            xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
            _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
            xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if xai_temperature_len > 0:
            S *= xai_temperature_reg

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
            )

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    kv_indptr_ptr,  # [num_seqs+1]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(kv_indptr_ptr + seq_idx + 1) - tl.load(kv_indptr_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * out_scale_inv
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def unified_attention(
    q,
    k,
    v,
    out,
    qo_indptr,
    max_seqlen_q,
    kv_indptr,
    kv_indices,
    softmax_scale,
    sliding_window_size=-1,
    softcap=0.0,
    k_scale=1.0,
    v_scale=1.0,
    custom_mask=None,
    mask_indptr=None,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    output_scale=None,
    sinks=None,
    xai_temperature_len=-1,
    enable_deterministic=False,
    is_causal=True,
    window_start_pos=None,
):
    """
    Args:
        q: [num_tokens, num_query_heads, head_size]
        k: [num_tokens, num_kv_heads, head_size] (page_size=1)
        v: [num_tokens, num_kv_heads, head_size] (page_size=1)
        out: [num_tokens, num_query_heads, head_size]
        qo_indptr: start of query [num_seqs+1]
        max_seqlen_q: max query length (>1: prefill, =1: decode)
        kv_indptr: start of kv [num_seqs+1]
        kv_indices: physical kv indices [total_kv_tokens]
        softmax_scale: softmax scale
        sliding_window_size: -1 for no sliding window
        softcap: 0.0 for no capping
        custom_mask: custom mask for spec decode
        mask_indptr: stores the index of the mask of sequences
        window_start_pos: absolute key start position per sequence for windowed kv
        seq_threshold_3D: threshold for kv split
        num_par_softmax_segments: number of parallel softmax segments for 3D kernel
        softmax_segm_output, softmax_segm_max, softmax_segm_expsum: intermediate buffer for 3D kernel
        output_scale: not available
    """
    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    num_seqs = len(kv_indptr) - 1
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    """
    BLOCK_M, TILE_SIZE, warps, num_stages can still be tuned
    """

    num_stages = None

    if (
        _is_cuda and CUDA_CAPABILITY[0] == 12
    ):  # sm120 has small smem and triton can have problem auto-selecting it
        num_stages = 2

    if max_seqlen_q == 1:  # decode
        BLOCK_M = max(8, num_queries_per_kv)
        num_warps = 4
    else:  # prefill
        BLOCK_M = 128
        num_warps = 8

    if _is_hip:  # reduce shared memory and register pressure for amd gpu
        BLOCK_M = min(64, BLOCK_M)
        num_warps = 4
        num_stages = 1

    TILE_SIZE_PREFILL = 64
    TILE_SIZE_DECODE = 64

    if head_size == 256:  # reduce shared memory pressure for gemma models
        TILE_SIZE_PREFILL = 32
        TILE_SIZE_DECODE = 32
        BLOCK_M = min(BLOCK_M, 64)

    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = sliding_window_size + 1 if sliding_window_size > 0 else 0

    if (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or enable_deterministic
        or custom_mask is not None
        or is_causal == False
    ):
        kernel_unified_attention_2d[
            (
                total_num_q_blocks,
                num_kv_heads,
            )
        ](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            kv_indptr_ptr=kv_indptr,
            kv_indices_ptr=kv_indices,
            scale=softmax_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            mask_ptr=custom_mask,
            mask_indptr=mask_indptr,
            window_start_pos_ptr=window_start_pos,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            TILE_SIZE=TILE_SIZE_PREFILL,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=sliding_window_val,
            stride_k_cache_t=k.stride(0),
            stride_k_cache_h=k.stride(1),
            stride_k_cache_d=k.stride(2),
            stride_v_cache_t=v.stride(0),
            stride_v_cache_h=v.stride(1),
            stride_v_cache_d=v.stride(2),
            query_start_len_ptr=qo_indptr,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
            num_warps=num_warps,
            USE_CUSTOM_MASK=custom_mask is not None,
            xai_temperature_len=xai_temperature_len,
            USE_WINDOW_START_POS=window_start_pos is not None,
            is_causal=is_causal,
            num_stages=num_stages,
        )
    else:
        kernel_unified_attention_3d[
            (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        ](
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            kv_indptr_ptr=kv_indptr,
            kv_indices_ptr=kv_indices,
            scale=softmax_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            softcap=softcap,
            window_start_pos_ptr=window_start_pos,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=sliding_window_val,
            stride_k_cache_t=k.stride(0),
            stride_k_cache_h=k.stride(1),
            stride_k_cache_d=k.stride(2),
            stride_v_cache_t=v.stride(0),
            stride_v_cache_h=v.stride(1),
            stride_v_cache_d=v.stride(2),
            query_start_len_ptr=qo_indptr,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            num_warps=num_warps,
            xai_temperature_len=xai_temperature_len,
            USE_WINDOW_START_POS=window_start_pos is not None,
            num_stages=num_stages,
        )
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            kv_indptr_ptr=kv_indptr,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=qo_indptr,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
            num_warps=num_warps,
        )
