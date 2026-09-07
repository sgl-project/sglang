# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from AITER's Triton unified-attention implementation.
"""MTP-verify specialization of AITER's Triton unified attention (3D split-K).

Forked from ROCm/aiter @ d9e5ef7, kernel_unified_attention_3d in
aiter/ops/triton/_triton_kernels/attention/unified_attention.py. The kernel body
is essentially unchanged; what diverges is the launch configuration, because
upstream's heuristics are tuned for prefill/decode and mis-fit MTP verify.

Divergences from upstream (upstream refs are in aiter/ops/triton/attention/
unified_attention.py):

  block_m = 32 -> BLOCK_Q = 2    upstream: BLOCK_M = 16 if nqpkv <= 16, BLOCK_Q = 1
  tile_size = 32, fixed          upstream: select_2d_config / select_3d_config
  always 3D split-K              upstream: use_2d_kernel() picks 2D or 3D
  num_segments computed here     upstream: config table
  num_warps / waves_per_eu       upstream: config table

The core win is block_m. MTP verify has q_len 2-4, so upstream packs one query
token x 16 heads per tile and fills only half the MFMA M dimension; block_m=32
stacks two draft tokens into one tile. Forcing 3D then recovers the CU occupancy
that a low batch would otherwise leave idle. Both are tied to the 16:1 GQA
ratio, not to head_dim.

Real constraints, mirrored by the gate in srt/layers/attention/aiter_backend.py:

  - head_dim must be a power of 2: HEAD_SIZE_PADDED is passed through unpadded,
    so 192 and 80 do not compile. 256 is the only measured size; 128 should be
    correct but is not tuned (num_warps=2 balances VGPR pressure at 256).
  - Causal only. seq_mask carries no tree mask, so spec decoding needs topk==1.

The gate is on shape, not on model; Qwen3.5-397B-A17B at TP1/TP2 (16:1 GQA,
head_dim 256) is the only config known to match today.
"""

import math

import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.types import e4m3_dtype

float8_info = torch.finfo(e4m3_dtype)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.math.exp2(Sdiv)
    p2 = tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


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


_unified_attention_3d_mtp_repr = make_kernel_repr(
    "unified_attention_3d_mtp",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "waves_per_eu",
        "num_stages",
        "ALL_DECODE",
        "SHUFFLED_KV_CACHE",
        "IS_Q_FP8",
        "IS_KV_FP8",
    ],
)


@triton.jit(repr=_unified_attention_3d_mtp_repr)
def unified_attention_3d_mtp_kernel(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    q_descale_ptr,  # float32
    k_descale_ptr,  # float32
    v_descale_ptr,  # float32
    out_scale_ptr,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    num_warps: tl.constexpr,  # int
    waves_per_eu: tl.constexpr,  # int
    num_stages: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    ALL_DECODE: tl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: tl.constexpr = False,  # bool
    K_WIDTH: tl.constexpr = 0,  # int
    IS_Q_FP8: tl.constexpr = False,  # bool
    IS_KV_FP8: tl.constexpr = False,  # bool
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    if ALL_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx: tl.int32 = 0
        cur_batch_query_len: tl.int32 = 1
        cur_batch_in_all_start_index: tl.int32 = q_block_global_idx
    else:
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
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    offs_shfl = None
    if SHUFFLED_KV_CACHE:
        offs_shfl = tl.arange(0, TILE_SIZE * HEAD_SIZE_PADDED)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            # Prescale with RCP_LN2, needed for exp2
            M = (
                tl.load(
                    sink_ptr + query_offset_1,
                    mask=query_mask_1,
                    other=float("-inf"),
                ).to(dtype=tl.float32)
                * RCP_LN2
            )
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

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

    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    if q_descale_ptr is not None:
        q_descale = tl.load(q_descale_ptr)
        qk_scale = qk_scale * q_descale
    else:
        q_descale = None

    if k_descale_ptr is not None:
        k_scale = tl.load(k_descale_ptr)
        qk_scale = qk_scale * k_scale
    else:
        k_scale = None

    out_factor: tl.float32 = 1.0
    if v_descale_ptr is not None:
        out_factor = tl.load(v_descale_ptr)

    if out_scale_ptr is not None:
        out_factor = out_factor / tl.load(out_scale_ptr)

    # iterate through tiles within current segment
    for j in range(
        segm_idx * tiles_per_segment,
        min((segm_idx + 1) * tiles_per_segment, num_tiles),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        k_mask = None
        v_mask = None
        other = None
        if SHUFFLED_KV_CACHE:
            physical_block_idx_shfl = tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64)
            k_offset = (
                physical_block_idx_shfl * stride_k_cache_0
                + kv_head_idx * stride_k_cache_1
                + offs_shfl
            )

            v_offset = (
                physical_block_idx_shfl * stride_v_cache_0
                + kv_head_idx * stride_v_cache_1
                + offs_shfl
            )
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            v_mask = dim_mask[None, :] & tile_mask[:, None]

            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            k_mask = dim_mask[:, None] & tile_mask[None, :]
            other = 0.0

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=k_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        K = K_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            K = (
                K.reshape(
                    HEAD_SIZE_PADDED // K_WIDTH,
                    TILE_SIZE,
                    K_WIDTH,
                )
                .permute(1, 0, 2)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
                .trans(1, 0)
            )

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=v_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        V = V_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            V = (
                V.reshape(
                    TILE_SIZE // K_WIDTH,
                    HEAD_SIZE_PADDED,
                    K_WIDTH,
                )
                .permute(0, 2, 1)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
            )

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_M, TILE_SIZE)
        # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
        S = qk_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            # prescale w. RCP_LN2 for later exp2
            S += alibi_slope[:, None] * (seq_offset - context_len) * RCP_LN2

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            # prescale w. RCP_LN2 for later exp2
            S += qq_bias * RCP_LN2

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    acc = acc * out_factor
    if NUM_SEGMENTS_PER_SEQ == 1:
        one_over_L = 1.0 / L[:, None]
        acc = acc * one_over_L

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
    if NUM_SEGMENTS_PER_SEQ > 1:
        segm_offset = (
            query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
            + query_offset_1 * NUM_SEGMENTS_PER_SEQ
            + segm_idx
        )
        tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
        tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


_unified_attention_3d_mtp_reduce_segments_repr = make_kernel_repr(
    "unified_attention_3d_mtp_reduce_segments",
    [
        "num_query_heads",
        "TILE_SIZE",
        "HEAD_SIZE",
        "NUM_SEGMENTS_PER_SEQ",
    ],
)


@triton.jit(repr=_unified_attention_3d_mtp_reduce_segments_repr)
def unified_attention_3d_mtp_reduce_segments_kernel(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_ptr,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    out_scale = None
    if out_scale_ptr is not None:
        out_scale = 1 / tl.load(out_scale_ptr)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)

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
    segm_expsum = segm_expsum * tl.math.exp2(segm_max - overall_max)
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
    segm_output *= tl.math.exp2(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if out_scale_ptr is not None:
        acc = acc * out_scale

    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(
        output_ptr + output_offset, acc.to(output_ptr.type.element_ty), mask=dim_mask
    )


def unified_attention_3d_mtp_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    block_table: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
) -> torch.Tensor:
    num_tokens, num_query_heads, head_size = q.shape
    num_seqs = seqused_k.shape[0]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads

    assert 1 < max_seqlen_q <= 4
    # Ratio, not absolute counts: block_m=32 // num_queries_per_kv gives block_q=2,
    # which packs two draft tokens per tile. Any (16N : N) GQA layout works because
    # the kernel indexes KV through kv_head_idx = program_id(1).
    assert num_query_heads % num_kv_heads == 0
    assert num_queries_per_kv == 16
    assert head_size == 256 and k.shape[1] == 16
    assert q.dtype == torch.bfloat16
    assert k.dtype == e4m3_dtype and v.dtype == e4m3_dtype

    block_m = 32
    block_q = block_m // num_queries_per_kv
    tile_size = 32
    total_num_q_blocks = num_tokens // block_q + num_seqs
    num_2d_programs = total_num_q_blocks * num_kv_heads

    num_cus = torch.cuda.get_device_properties(q.device).multi_processor_count
    max_segments = min(64, math.ceil(max_seqlen_k / tile_size))
    parallel_segments = math.ceil(num_cus * 2 / max(1, num_2d_programs))
    work_segments = math.ceil(max_seqlen_k / (tile_size * 32))
    work_segments = min(work_segments, parallel_segments * 8)
    num_segments = min(max_segments, max(8, parallel_segments, work_segments))
    num_segments = min(64, triton.next_power_of_2(num_segments))

    segment_output = torch.empty(
        num_tokens,
        num_query_heads,
        num_segments,
        head_size,
        dtype=torch.float32,
        device=q.device,
    )
    segment_max = torch.empty(
        num_tokens,
        num_query_heads,
        num_segments,
        dtype=torch.float32,
        device=q.device,
    )
    segment_expsum = torch.empty_like(segment_max)

    unified_attention_3d_mtp_kernel[(total_num_q_blocks, num_kv_heads, num_segments)](
        segm_output_ptr=segment_output,
        segm_max_ptr=segment_max,
        segm_expsum_ptr=segment_expsum,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=None,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=None,
        qq_bias_ptr=None,
        scale=softmax_scale,
        q_descale_ptr=None,
        k_descale_ptr=k_descale,
        v_descale_ptr=v_descale,
        out_scale_ptr=None,
        softcap=0.0,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        qq_bias_stride_0=0,
        BLOCK_SIZE=k.shape[1],
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size,
        USE_ALIBI_SLOPES=False,
        USE_QQ_BIAS=False,
        USE_SOFTCAP=False,
        USE_SINKS=False,
        SLIDING_WINDOW=0,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        num_seqs=num_seqs,
        BLOCK_M=block_m,
        ALL_DECODE=False,
        SHUFFLED_KV_CACHE=False,
        K_WIDTH=16,
        IS_Q_FP8=False,
        IS_KV_FP8=True,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        num_warps=2,
        waves_per_eu=2,
        num_stages=2,
    )

    unified_attention_3d_mtp_reduce_segments_kernel[(num_tokens, num_query_heads)](
        output_ptr=out,
        segm_output_ptr=segment_output,
        segm_max_ptr=segment_max,
        segm_expsum_ptr=segment_expsum,
        seq_lens_ptr=seqused_k,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        out_scale_ptr=None,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=block_table.stride(0),
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        num_warps=2,
        waves_per_eu=2,
        num_stages=1,
    )
    return out
