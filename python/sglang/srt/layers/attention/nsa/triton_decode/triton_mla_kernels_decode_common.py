"""
Common utilities and attention kernels for Triton MLA Decode.

This module contains shared code for the DeepSeek V4 Triton decode implementation:
- Attention kernels (unified sparse decode)
- Helper functions for chunked attention
- Token range computation for memory-based chunking
"""

from typing import List, Tuple

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


# ============================================================================
# Bucketing for autotune keys to avoid recompilation per unique batch size
# ============================================================================
def _bucket_total_tokens(total_tokens: int) -> int:
    """Round total_tokens up to the nearest power of 2 for autotune key stability.

    In serving, total_tokens (= batch_size * seq_len) varies with every batch.
    Using the exact value as an autotune key causes recompilation for each unique
    value. Bucketing to powers of 2 limits the number of unique keys to ~15,
    dramatically reducing autotuning overhead.

    Returns:
        Power-of-2 bucket: 1, 2, 4, 8, ..., up to the next power of 2.
    """
    if total_tokens <= 0:
        return 1
    # Round up to next power of 2
    n = 1
    while n < total_tokens:
        n <<= 1
    return n


# ============================================================================
# Helper function to compute workload size category for autotune
# ============================================================================
def _get_workload_size_category(total_tokens: int, topk: int) -> int:
    """
    Compute workload size category for autotune key.
    Returns:
        0: small (< 10K elements)
        1: medium (10K - 100K elements)
        2: large (100K - 1M elements)
        3: very large (> 1M elements)
    """
    total_elements = total_tokens * topk
    if total_elements < 10000:
        return 0
    elif total_elements < 100000:
        return 1
    elif total_elements < 1000000:
        return 2
    else:
        return 3


# ============================================================================
# Unified Attention Kernels
# ============================================================================


# ============================================================================
# CDNA4 (gfx950) Optimized: Added high-performance configs for MI355X
# Best config for h_q=128, large topk: BLOCK_H=64, BLOCK_N=256, num_warps=8
# ============================================================================
@triton.autotune(
    configs=[
        # CDNA4-optimized configurations for MI355X (256 CUs, 8TB/s BW)
        # Best for h_q=128, large topk: maximize parallelism across CUs
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 128, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 128, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        # Additional configs with num_stages=2 for better pipelining
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 64, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=2
        ),
        # Original configurations
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 64, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 16, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 32, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 8, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_H": 8, "BLOCK_N": 512, "BLOCK_D": 128}, num_warps=8, num_stages=1
        ),
    ],
    key=["total_tokens_bucket", "h_q", "total_topk", "d_qk"],
)
@triton.jit
def _unified_sparse_decode_kernel(
    Q,
    KV,
    Mask,
    AttnSink,
    Output,
    LSE,
    sm_scale,
    total_tokens,
    total_tokens_bucket,
    h_q,
    total_topk,
    d_qk,
    d_v,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_t,
    stride_kv_k,
    stride_kv_d,
    stride_mask_t,
    stride_mask_k,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_t,
    stride_lse_h,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Unified attention kernel with single KV buffer (int64 safe)."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    NEG_INF = float("-inf")
    POS_INF = float("+inf")

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64)
    stride_kv_t_64 = tl.cast(stride_kv_t, tl.int64)
    stride_mask_t_64 = tl.cast(stride_mask_t, tl.int64)
    q_base = Q + pid_t_64 * stride_q_t_64
    kv_base = KV + pid_t_64 * stride_kv_t_64
    mask_base = Mask + pid_t_64 * stride_mask_t_64

    for n_start in range(0, total_topk, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < total_topk

        mask_ptrs = mask_base + offs_n * stride_mask_k
        invalid = tl.load(mask_ptrs, mask=mask_n, other=True)
        valid = mask_n & ~invalid

        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)

        for d_start in range(0, d_qk, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < d_qk

            q_ptrs = (
                q_base + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d
            )
            q_chunk = tl.load(
                q_ptrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0
            ).to(tl.bfloat16)

            k_ptrs = (
                kv_base + offs_n[:, None] * stride_kv_k + offs_d[None, :] * stride_kv_d
            )
            k_chunk = tl.load(
                k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0
            ).to(tl.bfloat16)

            qk += tl.dot(q_chunk, tl.trans(k_chunk))

        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, NEG_INF)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
        p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
        l_new = alpha * l_i + tl.sum(p, axis=1)
        p_bf16 = p.to(tl.bfloat16)

        offs_v = tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, v)

        offs_v = BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0
        ).to(tl.bfloat16)
        acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, v)

        offs_v = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0
        ).to(tl.bfloat16)
        acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, v)

        offs_v = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = kv_base + offs_n[:, None] * stride_kv_k + offs_v[None, :] * stride_kv_d
        v = tl.load(
            v_ptrs, mask=valid[:, None] & (offs_v[None, :] < d_v), other=0.0
        ).to(tl.bfloat16)
        acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, v)

        m_i = m_new
        l_i = l_new

    lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
    is_lonely_q = l_i == 0.0

    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)
        denominator = l_i + exp_attn_sink_minus_m
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

    # Pre-compute 2D versions for efficiency
    is_lonely_q_2d = is_lonely_q[:, None]
    output_scale_2d = output_scale[:, None]
    acc_0 = tl.where(is_lonely_q_2d, 0.0, acc_0 * output_scale_2d)
    acc_1 = tl.where(is_lonely_q_2d, 0.0, acc_1 * output_scale_2d)
    acc_2 = tl.where(is_lonely_q_2d, 0.0, acc_2 * output_scale_2d)
    acc_3 = tl.where(is_lonely_q_2d, 0.0, acc_3 * output_scale_2d)
    lse = tl.where(is_lonely_q, POS_INF, lse)

    stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)
    tl.store(LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h, lse, mask=mask_h)

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64
    # Pre-compute 2D versions
    offs_h_2d = offs_h[:, None]
    mask_h_2d = mask_h[:, None]
    offs_v_0 = tl.arange(0, BLOCK_D)
    offs_v_1 = BLOCK_D + tl.arange(0, BLOCK_D)
    offs_v_2 = 2 * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_v_3 = 3 * BLOCK_D + tl.arange(0, BLOCK_D)
    tl.store(
        o_base + offs_h_2d * stride_o_h + offs_v_0[None, :] * stride_o_d,
        acc_0.to(tl.bfloat16),
        mask=mask_h_2d,
    )
    tl.store(
        o_base + offs_h_2d * stride_o_h + offs_v_1[None, :] * stride_o_d,
        acc_1.to(tl.bfloat16),
        mask=mask_h_2d & (offs_v_1[None, :] < d_v),
    )
    tl.store(
        o_base + offs_h_2d * stride_o_h + offs_v_2[None, :] * stride_o_d,
        acc_2.to(tl.bfloat16),
        mask=mask_h_2d & (offs_v_2[None, :] < d_v),
    )
    tl.store(
        o_base + offs_h_2d * stride_o_h + offs_v_3[None, :] * stride_o_d,
        acc_3.to(tl.bfloat16),
        mask=mask_h_2d & (offs_v_3[None, :] < d_v),
    )


# ============================================================================
# Attention Runner Functions
# ============================================================================


def run_unified_attention(
    q_reshaped,
    gathered_kv,
    invalid_mask,
    d_v,
    sm_scale,
    total_tokens,
    h_q,
    total_topk,
    d_qk,
    attn_sink=None,
):
    """Run unified attention with single KV buffer.

    Run unified sparse decode attention kernel.
    """
    output = torch.empty(
        (total_tokens, h_q, d_v), dtype=torch.bfloat16, device=q_reshaped.device
    )
    lse = torch.empty(
        (total_tokens, h_q), dtype=torch.float32, device=q_reshaped.device
    )

    HAS_ATTN_SINK = attn_sink is not None
    attn_sink_tensor = attn_sink if HAS_ATTN_SINK else lse[:1]

    grid = lambda meta: (total_tokens, triton.cdiv(h_q, meta["BLOCK_H"]))
    _unified_sparse_decode_kernel[grid](
        q_reshaped,
        gathered_kv,
        invalid_mask,
        attn_sink_tensor,
        output,
        lse,
        sm_scale,
        total_tokens,
        _bucket_total_tokens(total_tokens),
        h_q,
        total_topk,
        d_qk,
        d_v,
        q_reshaped.stride(0),
        q_reshaped.stride(1),
        q_reshaped.stride(2),
        gathered_kv.stride(0),
        gathered_kv.stride(1),
        gathered_kv.stride(2),
        invalid_mask.stride(0),
        invalid_mask.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        HAS_ATTN_SINK=HAS_ATTN_SINK,
    )
    return output, lse


def run_chunked_attention_triton(
    q_reshaped,
    gathered_kv,
    invalid_mask,
    d_v,
    sm_scale,
    total_tokens,
    h_q,
    total_topk,
    d_qk,
    attn_sink=None,
    chunk_size=8192,
):
    """Chunked attention using Triton kernels with cross-chunk softmax merging."""
    device = q_reshaped.device

    num_chunks = (total_topk + chunk_size - 1) // chunk_size

    kv_chunks = []
    mask_chunks = []
    chunk_sizes = []

    for chunk_idx in range(num_chunks):
        start_k = chunk_idx * chunk_size
        end_k = min(start_k + chunk_size, total_topk)
        chunk_topk = end_k - start_k
        chunk_sizes.append(chunk_topk)
        kv_chunks.append(gathered_kv[:, start_k:end_k, :].contiguous())
        mask_chunks.append(invalid_mask[:, start_k:end_k].contiguous())

    lse_acc = torch.full(
        (total_tokens, h_q), float("-inf"), dtype=torch.float32, device=device
    )
    acc = torch.zeros((total_tokens, h_q, d_v), dtype=torch.float32, device=device)

    for chunk_idx in range(num_chunks):
        kv_chunk = kv_chunks[chunk_idx]
        mask_chunk = mask_chunks[chunk_idx]
        chunk_topk = chunk_sizes[chunk_idx]

        chunk_output, chunk_lse = run_unified_attention(
            q_reshaped,
            kv_chunk,
            mask_chunk,
            d_v,
            sm_scale,
            total_tokens,
            h_q,
            chunk_topk,
            d_qk,
            attn_sink=None,
        )

        is_chunk_lonely = torch.isinf(chunk_lse) & (chunk_lse > 0)

        chunk_lse_for_merge = torch.where(
            is_chunk_lonely, torch.full_like(chunk_lse, float("-inf")), chunk_lse
        )

        lse_max = torch.maximum(lse_acc, chunk_lse_for_merge)

        exp_acc = torch.exp(lse_acc - lse_max)
        exp_acc = torch.where(torch.isnan(exp_acc), torch.zeros_like(exp_acc), exp_acc)

        exp_chunk = torch.exp(chunk_lse_for_merge - lse_max)
        exp_chunk = torch.where(
            torch.isnan(exp_chunk) | is_chunk_lonely,
            torch.zeros_like(exp_chunk),
            exp_chunk,
        )

        sum_exp = exp_acc + exp_chunk
        lse_new = lse_max + torch.log(
            torch.where(sum_exp == 0, torch.ones_like(sum_exp), sum_exp)
        )

        both_empty = (lse_acc == float("-inf")) & (chunk_lse_for_merge == float("-inf"))
        lse_new = torch.where(
            both_empty, torch.full_like(lse_new, float("-inf")), lse_new
        )

        weight_acc = torch.exp(lse_acc - lse_new)
        weight_acc = torch.where(
            torch.isnan(weight_acc) | torch.isinf(weight_acc),
            torch.zeros_like(weight_acc),
            weight_acc,
        )

        weight_chunk = torch.exp(chunk_lse_for_merge - lse_new)
        weight_chunk = torch.where(
            torch.isnan(weight_chunk) | torch.isinf(weight_chunk) | is_chunk_lonely,
            torch.zeros_like(weight_chunk),
            weight_chunk,
        )

        acc = (
            weight_acc.unsqueeze(-1) * acc
            + weight_chunk.unsqueeze(-1) * chunk_output.float()
        )

        lse_acc = lse_new

    output = acc
    lse = lse_acc

    is_lonely_final = lse == float("-inf")

    lse = torch.where(is_lonely_final, torch.full_like(lse, float("+inf")), lse)

    if attn_sink is not None:
        attn_sink_expanded = attn_sink.view(1, h_q)
        exp_diff = torch.exp(attn_sink_expanded - lse)
        exp_diff = torch.where(
            is_lonely_final, torch.full_like(exp_diff, float("inf")), exp_diff
        )
        scale = 1.0 / (1.0 + exp_diff)
        output = output * scale.unsqueeze(-1)

    output = torch.where(
        is_lonely_final.unsqueeze(-1), torch.zeros_like(output), output
    )

    return output.to(torch.bfloat16), lse


# ============================================================================
# Helper class and functions for token-range based chunking
# ============================================================================


class SlicedKVScope:
    """A sliced view of KV scope for a specific token range."""

    __slots__ = [
        "blocked_k",
        "blocked_k_quantized",
        "indices_in_kvcache",
        "topk_length",
    ]

    def __init__(self, blocked_k, blocked_k_quantized, indices_in_kvcache, topk_length):
        self.blocked_k = blocked_k
        self.blocked_k_quantized = blocked_k_quantized
        self.indices_in_kvcache = indices_in_kvcache
        self.topk_length = topk_length


def slice_kv_scope_for_tokens(orig_scope, start_t: int, end_t: int, s_q: int):
    """Slice a KV scope to only include tokens in range [start_t, end_t)."""
    if orig_scope is None:
        return None

    orig_indices = orig_scope.indices_in_kvcache.reshape(
        -1, orig_scope.indices_in_kvcache.size(-1)
    )
    sliced_indices = orig_indices[start_t:end_t]

    sliced_topk_length = None
    if orig_scope.topk_length is not None:
        batch_start = start_t // s_q
        batch_end = (end_t + s_q - 1) // s_q
        batch_topk_length = orig_scope.topk_length[batch_start:batch_end]
        if s_q > 1:
            chunk_tokens = end_t - start_t
            expanded = batch_topk_length.unsqueeze(1).expand(-1, s_q).reshape(-1)
            offset_in_first_batch = start_t % s_q
            sliced_topk_length = expanded[
                offset_in_first_batch : offset_in_first_batch + chunk_tokens
            ]
        else:
            sliced_topk_length = batch_topk_length

    return SlicedKVScope(
        blocked_k=orig_scope.blocked_k,
        blocked_k_quantized=orig_scope.blocked_k_quantized,
        indices_in_kvcache=sliced_indices,
        topk_length=sliced_topk_length,
    )


def compute_token_ranges(
    total_tokens: int,
    total_topk: int,
    d_qk: int,
    max_buffer_bytes: int = 2 * 1024 * 1024 * 1024,
) -> List[Tuple[int, int]]:
    """Compute token ranges for processing, chunking if buffer would exceed limit."""
    buffer_size_bytes = total_tokens * total_topk * d_qk * 2

    if buffer_size_bytes <= max_buffer_bytes:
        return [(0, total_tokens)]

    max_tokens_per_chunk = max_buffer_bytes // (total_topk * d_qk * 2)
    chunk_size = max(1, max_tokens_per_chunk)

    token_ranges = []
    start_t = 0
    while start_t < total_tokens:
        end_t = min(start_t + chunk_size, total_tokens)
        token_ranges.append((start_t, end_t))
        start_t = end_t

    return token_ranges


# ============================================================================
# Split-K Attention for Large TopK
# ============================================================================
def run_splitk_unified_attention(
    q_reshaped,
    gathered_kv,
    invalid_mask,
    d_v,
    sm_scale,
    total_tokens,
    h_q,
    total_topk,
    d_qk,
    attn_sink=None,
    split_k=4,
):
    """Run split-K attention for large topk cases."""
    from .triton_mla_kernels_decode_splitk import run_splitk_attention

    return run_splitk_attention(
        q_reshaped,
        gathered_kv,
        invalid_mask,
        d_v,
        sm_scale,
        total_tokens,
        h_q,
        total_topk,
        d_qk,
        attn_sink=attn_sink,
        split_k=split_k,
    )
