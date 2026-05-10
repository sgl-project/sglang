"""
Fused Gather+Dequant+Attention Kernel for DSV4 (d_qk=512)

This module implements a fused kernel that combines:
1. Gather: Load KV from sparse indices
2. Dequant: FP8 to BF16 dequantization
3. Attention: Compute attention scores and output

Benefits for workloads without extra scope:
- Eliminates intermediate buffer (gathered_kv) write/read
- Reduces kernel launch overhead (1 kernel instead of 2)
- Better cache utilization

Supports:
- DSV4 (d_qk=512): 7 tiles of 64, uint8 scales
- All configs: with/without topk_length, with/without attn_sink

OPTIMIZED VERSION: Reduced code duplication in dual-scope kernel by using
a helper function for KV block processing.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .triton_mla_kernels_decode_common import _bucket_total_tokens

# ============================================================================
# Constants for DSV4 layout
# ============================================================================
DSV4_D_QK = 512
DSV4_D_NOPE = 448
DSV4_D_ROPE = 64
DSV4_D_V = 512
DSV4_TILE_SIZE = 64
DSV4_NUM_TILES = 7
DSV4_BYTES_PER_TOKEN_DATA = 576  # 448 nope + 128 rope
DSV4_BYTES_PER_TOKEN_SCALE = 8  # 7 scales + 1 padding


# ============================================================================
# Helper: Process KV block and compute QK scores + accumulator update
# This is the core computation shared by both single and dual scope kernels
# ============================================================================
@triton.jit
def _process_kv_block_aggressive(
    # KV cache parameters
    kv_block_base,
    nope_rope_offset,
    scale_base_offset,
    valid,
    valid_2d,
    # Query tiles
    q_0,
    q_1,
    q_2,
    q_3,
    q_4,
    q_5,
    q_6,
    q_7,
    # Accumulators (passed by reference via return)
    acc_0,
    acc_1,
    acc_2,
    acc_3,
    acc_4,
    acc_5,
    acc_6,
    acc_7,
    # Softmax state
    m_i,
    l_i,
    # Other parameters
    offs_tile,
    sm_scale,
    # Constants
    TILE_SIZE: tl.constexpr,
    D_NOPE: tl.constexpr,
    LOG2E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Process one block of KV tokens with batch loading.
    Key optimization: Load all KV tiles first, then process them.
    """
    NEG_INF = float("-inf")

    scale_ptrs = kv_block_base + scale_base_offset
    scale_uint8_0 = tl.load(scale_ptrs, mask=valid, other=127).to(tl.uint8)
    scale_uint8_1 = tl.load(scale_ptrs + 1, mask=valid, other=127).to(tl.uint8)
    scale_uint8_2 = tl.load(scale_ptrs + 2, mask=valid, other=127).to(tl.uint8)
    scale_uint8_3 = tl.load(scale_ptrs + 3, mask=valid, other=127).to(tl.uint8)
    scale_uint8_4 = tl.load(scale_ptrs + 4, mask=valid, other=127).to(tl.uint8)
    scale_uint8_5 = tl.load(scale_ptrs + 5, mask=valid, other=127).to(tl.uint8)
    scale_uint8_6 = tl.load(scale_ptrs + 6, mask=valid, other=127).to(tl.uint8)

    tile_base = kv_block_base[:, None] + nope_rope_offset[:, None]

    # Batch load all tiles
    nope_uint8_0 = tl.load(tile_base + offs_tile[None, :], mask=valid_2d, other=0)
    nope_uint8_1 = tl.load(
        tile_base + TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    nope_uint8_2 = tl.load(
        tile_base + 2 * TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    nope_uint8_3 = tl.load(
        tile_base + 3 * TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    nope_uint8_4 = tl.load(
        tile_base + 4 * TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    nope_uint8_5 = tl.load(
        tile_base + 5 * TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    nope_uint8_6 = tl.load(
        tile_base + 6 * TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    rope_ptrs = tile_base + D_NOPE + offs_tile[None, :] * 2
    rope_lo = tl.load(rope_ptrs, mask=valid_2d, other=0).to(tl.uint16)
    rope_hi = tl.load(rope_ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)

    scale_bf16_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_4 = tl.math.exp2(scale_uint8_4.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_5 = tl.math.exp2(scale_uint8_5.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_6 = tl.math.exp2(scale_uint8_6.to(tl.float32) - 127.0).to(tl.bfloat16)

    qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)

    nope_fp8_0 = nope_uint8_0.to(tl.float8e4nv, bitcast=True)
    kv_0 = (nope_fp8_0.to(tl.bfloat16) * scale_bf16_0[:, None]).to(tl.bfloat16)
    kv_0 = tl.where(valid_2d, kv_0, 0.0)
    qk += tl.dot(q_0, tl.trans(kv_0)).to(tl.float32)

    nope_fp8_1 = nope_uint8_1.to(tl.float8e4nv, bitcast=True)
    kv_1 = (nope_fp8_1.to(tl.bfloat16) * scale_bf16_1[:, None]).to(tl.bfloat16)
    kv_1 = tl.where(valid_2d, kv_1, 0.0)
    qk += tl.dot(q_1, tl.trans(kv_1)).to(tl.float32)

    nope_fp8_2 = nope_uint8_2.to(tl.float8e4nv, bitcast=True)
    kv_2 = (nope_fp8_2.to(tl.bfloat16) * scale_bf16_2[:, None]).to(tl.bfloat16)
    kv_2 = tl.where(valid_2d, kv_2, 0.0)
    qk += tl.dot(q_2, tl.trans(kv_2)).to(tl.float32)

    nope_fp8_3 = nope_uint8_3.to(tl.float8e4nv, bitcast=True)
    kv_3 = (nope_fp8_3.to(tl.bfloat16) * scale_bf16_3[:, None]).to(tl.bfloat16)
    kv_3 = tl.where(valid_2d, kv_3, 0.0)
    qk += tl.dot(q_3, tl.trans(kv_3)).to(tl.float32)

    nope_fp8_4 = nope_uint8_4.to(tl.float8e4nv, bitcast=True)
    kv_4 = (nope_fp8_4.to(tl.bfloat16) * scale_bf16_4[:, None]).to(tl.bfloat16)
    kv_4 = tl.where(valid_2d, kv_4, 0.0)
    qk += tl.dot(q_4, tl.trans(kv_4)).to(tl.float32)

    nope_fp8_5 = nope_uint8_5.to(tl.float8e4nv, bitcast=True)
    kv_5 = (nope_fp8_5.to(tl.bfloat16) * scale_bf16_5[:, None]).to(tl.bfloat16)
    kv_5 = tl.where(valid_2d, kv_5, 0.0)
    qk += tl.dot(q_5, tl.trans(kv_5)).to(tl.float32)

    nope_fp8_6 = nope_uint8_6.to(tl.float8e4nv, bitcast=True)
    kv_6 = (nope_fp8_6.to(tl.bfloat16) * scale_bf16_6[:, None]).to(tl.bfloat16)
    kv_6 = tl.where(valid_2d, kv_6, 0.0)
    qk += tl.dot(q_6, tl.trans(kv_6)).to(tl.float32)

    kv_7 = (rope_lo | (rope_hi << 8)).to(tl.bfloat16, bitcast=True)
    kv_7 = tl.where(valid_2d, kv_7, 0.0)
    qk += tl.dot(q_7, tl.trans(kv_7)).to(tl.float32)

    qk = qk * sm_scale
    qk = tl.where(valid[None, :], qk, NEG_INF)

    m_ij = tl.max(qk, axis=1)
    m_new = tl.maximum(m_i, m_ij)
    alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
    p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
    l_new = alpha * l_i + tl.sum(p, axis=1)
    p_bf16 = p.to(tl.bfloat16)

    acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, kv_0).to(tl.float32)
    acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, kv_1).to(tl.float32)
    acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, kv_2).to(tl.float32)
    acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, kv_3).to(tl.float32)
    acc_4 = acc_4 * alpha[:, None] + tl.dot(p_bf16, kv_4).to(tl.float32)
    acc_5 = acc_5 * alpha[:, None] + tl.dot(p_bf16, kv_5).to(tl.float32)
    acc_6 = acc_6 * alpha[:, None] + tl.dot(p_bf16, kv_6).to(tl.float32)
    acc_7 = acc_7 * alpha[:, None] + tl.dot(p_bf16, kv_7).to(tl.float32)

    return acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_new, l_new


# ============================================================================
# DSV4 Fused Gather+Dequant+Attention Kernel (Single Scope)
# ============================================================================
@triton.autotune(
    configs=[
        # Removed 3 exact duplicates
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 256}, num_warps=4, num_stages=1),
    ],
    key=["total_tokens_bucket", "h_q", "topk"],
)
@triton.jit
def _fused_gather_attn_dsv4_kernel(
    Q,
    KV_Cache,
    Indices,
    TopkLength,
    AttnSink,
    Output,
    LSE,
    sm_scale,
    total_tokens,
    total_tokens_bucket,
    h_q,
    topk,
    num_blocks,
    block_size,
    s_q,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_block,
    stride_idx_t,
    stride_idx_k,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_t,
    stride_lse_h,
    HAS_TOPK_LENGTH: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused gather+dequant+attention kernel for DSV4."""
    LOG2E: tl.constexpr = 1.4426950408889634
    D_NOPE: tl.constexpr = 448
    D_ROPE: tl.constexpr = 64
    TILE_SIZE: tl.constexpr = 64
    BYTES_PER_TOKEN_DATA: tl.constexpr = 576
    BYTES_PER_TOKEN_SCALE: tl.constexpr = 8

    # OPTIMIZED: Swapped grid - pid_h first for better cache locality
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    NEG_INF = float("-inf")

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_4 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_5 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_6 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_7 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64)
    q_base = Q + pid_t_64 * stride_q_t_64

    batch_idx = pid_t // s_q
    offs_tile = tl.arange(0, TILE_SIZE)

    q_0 = tl.load(
        q_base + offs_h[:, None] * stride_q_h + offs_tile[None, :] * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_1 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_2 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (2 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_3 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (3 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_4 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (4 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_5 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (5 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_6 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (6 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_7 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (7 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    # Early-exit: pre-load topk_len and skip invalid blocks
    if HAS_TOPK_LENGTH:
        topk_len = tl.load(TopkLength + batch_idx)

    for n_start in range(0, topk, BLOCK_N):
        # Skip entire block if beyond valid topk range
        should_compute = not HAS_TOPK_LENGTH or n_start < topk_len
        if should_compute:
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < topk

            idx_ptrs = Indices + pid_t * stride_idx_t + offs_n * stride_idx_k
            indices = tl.load(idx_ptrs, mask=mask_n, other=-1)

            is_invalid = indices == -1
            if HAS_TOPK_LENGTH:
                is_invalid = is_invalid | (offs_n >= topk_len)

            valid = mask_n & ~is_invalid
            indices_clamped = tl.maximum(indices, 0)

            block_idx = indices_clamped // block_size
            offset_in_block = indices_clamped % block_size

            block_idx_64 = block_idx.to(tl.int64)
            offset_in_block_64 = offset_in_block.to(tl.int64)

            stride_kv_block_64 = tl.cast(stride_kv_block, tl.int64)
            kv_block_base = KV_Cache + block_idx_64 * stride_kv_block_64
            nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
            scale_base_offset = (
                block_size * BYTES_PER_TOKEN_DATA
                + offset_in_block_64 * BYTES_PER_TOKEN_SCALE
            )

            valid_2d = valid[:, None]

            # Use helper function for KV processing
            acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = (
                _process_kv_block_aggressive(
                    kv_block_base,
                    nope_rope_offset,
                    scale_base_offset,
                    valid,
                    valid_2d,
                    q_0,
                    q_1,
                    q_2,
                    q_3,
                    q_4,
                    q_5,
                    q_6,
                    q_7,
                    acc_0,
                    acc_1,
                    acc_2,
                    acc_3,
                    acc_4,
                    acc_5,
                    acc_6,
                    acc_7,
                    m_i,
                    l_i,
                    offs_tile,
                    sm_scale,
                    TILE_SIZE,
                    D_NOPE,
                    LOG2E,
                    BLOCK_H,
                    BLOCK_N,
                )
            )

    # Finalize
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

    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0 * output_scale[:, None])
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1 * output_scale[:, None])
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2 * output_scale[:, None])
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3 * output_scale[:, None])
    acc_4 = tl.where(is_lonely_q[:, None], 0.0, acc_4 * output_scale[:, None])
    acc_5 = tl.where(is_lonely_q[:, None], 0.0, acc_5 * output_scale[:, None])
    acc_6 = tl.where(is_lonely_q[:, None], 0.0, acc_6 * output_scale[:, None])
    acc_7 = tl.where(is_lonely_q[:, None], 0.0, acc_7 * output_scale[:, None])
    lse = tl.where(is_lonely_q, float("+inf"), lse)

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64

    # Optimized output stores with pre-computed row base pointers
    # Convert to bfloat16 first (batch conversion)
    o_0 = acc_0.to(tl.bfloat16)
    o_1 = acc_1.to(tl.bfloat16)
    o_2 = acc_2.to(tl.bfloat16)
    o_3 = acc_3.to(tl.bfloat16)
    o_4 = acc_4.to(tl.bfloat16)
    o_5 = acc_5.to(tl.bfloat16)
    o_6 = acc_6.to(tl.bfloat16)
    o_7 = acc_7.to(tl.bfloat16)

    # Pre-compute row base pointers (shared across all 8 stores)
    row_ptrs = o_base + offs_h[:, None] * stride_o_h

    # Store all 8 tiles with optimized pointer arithmetic
    tl.store(row_ptrs + offs_tile[None, :] * stride_o_d, o_0, mask=mask_h[:, None])
    tl.store(
        row_ptrs + (TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_1,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (2 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_2,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (3 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_3,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (4 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_4,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (5 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_5,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (6 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_6,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (7 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_7,
        mask=mask_h[:, None],
    )

    lse_ptrs = LSE + pid_t * stride_lse_t + offs_h * stride_lse_h
    tl.store(lse_ptrs, lse, mask=mask_h)


# Threshold for disabling AMD buffer_ops optimization
# When KV cache size exceeds INT32_MAX, buffer_ops can cause int32 overflow
# INT32_MAX = 2^31 - 1 = 2,147,483,647 bytes (~2GB)
BUFFER_OPS_DISABLE_THRESHOLD = 2 * 1024 * 1024 * 1024  # 2GB


def fused_gather_attn_decode_dsv4(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    sm_scale: float,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    s_q: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused gather+dequant+attention for DSV4.
    Uses Split-K optimization for large topk (>= 8192).

    Args:
        q: Query tensor [total_tokens, h_q, d_qk]
        kv_cache: Quantized KV cache
        indices: KV indices [total_tokens, topk]
        block_size: Block size for KV cache
        sm_scale: Softmax scale
        topk_length: Optional per-batch topk length [b]
        attn_sink: Optional attention sink values [h_q]
        s_q: Sequence length per batch

    Returns:
        output: Attention output [total_tokens, h_q, d_v]
        lse: Log-sum-exp values [total_tokens, h_q]
    """
    total_tokens, h_q, d_qk = q.shape
    topk = indices.shape[1]
    d_v = DSV4_D_V
    device = q.device

    kv_uint8 = kv_cache.view(torch.uint8)
    num_blocks = kv_cache.shape[0]
    stride_kv_block = kv_uint8.stride(0)
    kv_flat = kv_uint8.reshape(num_blocks, -1)

    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        q = q.to(torch.bfloat16).contiguous()

    if not indices.is_contiguous():
        indices = indices.contiguous()

    kv_cache_size = stride_kv_block * num_blocks
    disable_buffer_ops = kv_cache_size > BUFFER_OPS_DISABLE_THRESHOLD

    # Use Split-K for large topk
    if topk >= SPLITK_TOPK_THRESHOLD:
        split_k = _select_split_k(topk, h_q, total_tokens)
        topk_per_split = (topk + split_k - 1) // split_k

        partial_output = torch.empty(
            split_k, total_tokens, h_q, d_v, dtype=torch.float32, device=device
        )
        partial_lse = torch.empty(
            split_k, total_tokens, h_q, dtype=torch.float32, device=device
        )
        output = torch.empty(
            total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device
        )
        lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

        topk_length_tensor = topk_length if topk_length is not None else lse[:1, 0]
        attn_sink_tensor = attn_sink if attn_sink is not None else lse[0, :]

        # Use autotuned grid
        grid_splitk = lambda meta: (
            triton.cdiv(h_q, meta["BLOCK_H"]),
            total_tokens,
            split_k,
        )

        def run_splitk_kernel():
            _fused_gather_attn_dsv4_splitk_kernel[grid_splitk](
                q,
                kv_flat,
                indices,
                topk_length_tensor,
                partial_output,
                partial_lse,
                sm_scale,
                total_tokens,
                _bucket_total_tokens(total_tokens),
                h_q,
                topk,
                num_blocks,
                block_size,
                s_q,
                topk_per_split,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                stride_kv_block,
                indices.stride(0),
                indices.stride(1),
                partial_output.stride(0),
                partial_output.stride(1),
                partial_output.stride(2),
                partial_output.stride(3),
                partial_lse.stride(0),
                partial_lse.stride(1),
                partial_lse.stride(2),
                HAS_TOPK_LENGTH=topk_length is not None,
            )

        if disable_buffer_ops:
            with triton.knobs.amd.scope():
                triton.knobs.amd.use_buffer_ops = False
                run_splitk_kernel()
        else:
            run_splitk_kernel()

        # Use autotuned combine kernel for split_k=8
        if split_k == 8:
            # Autotuned kernel - grid is determined by autotune
            grid_combine = lambda meta: (
                total_tokens,
                triton.cdiv(h_q, meta["BLOCK_H"]),
            )
            _combine_splitk_kernel_8_optimized[grid_combine](
                partial_output,
                partial_lse,
                attn_sink_tensor,
                output,
                lse,
                total_tokens,
                _bucket_total_tokens(total_tokens),
                h_q,
                d_v,
                partial_output.stride(0),
                partial_output.stride(1),
                partial_output.stride(2),
                partial_output.stride(3),
                partial_lse.stride(0),
                partial_lse.stride(1),
                partial_lse.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                lse.stride(0),
                lse.stride(1),
                HAS_ATTN_SINK=attn_sink is not None,
            )
        else:
            BLOCK_H_COMBINE = 16
            BLOCK_D_COMBINE = 128
            grid_combine = (total_tokens, triton.cdiv(h_q, BLOCK_H_COMBINE))

            # Select appropriate combine kernel based on split_k
            if split_k == 2:
                combine_kernel = _combine_splitk_kernel_2
            elif split_k == 4:
                combine_kernel = _combine_splitk_kernel
            else:
                raise ValueError(f"Unsupported split_k: {split_k}")

            combine_kernel[grid_combine](
                partial_output,
                partial_lse,
                attn_sink_tensor,
                output,
                lse,
                total_tokens,
                _bucket_total_tokens(total_tokens),
                h_q,
                d_v,
                partial_output.stride(0),
                partial_output.stride(1),
                partial_output.stride(2),
                partial_output.stride(3),
                partial_lse.stride(0),
                partial_lse.stride(1),
                partial_lse.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                lse.stride(0),
                lse.stride(1),
                HAS_ATTN_SINK=attn_sink is not None,
                BLOCK_H=BLOCK_H_COMBINE,
                BLOCK_D=BLOCK_D_COMBINE,
                num_warps=4,
                num_stages=1,
            )

        return output, lse

    # Use original kernel for smaller topk
    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

    topk_length_tensor = topk_length if topk_length is not None else lse[:1, 0]
    attn_sink_tensor = attn_sink if attn_sink is not None else lse[0, :]

    grid = lambda meta: (triton.cdiv(h_q, meta["BLOCK_H"]), total_tokens)

    def run_kernel():
        _fused_gather_attn_dsv4_kernel[grid](
            q,
            kv_flat,
            indices,
            topk_length_tensor,
            attn_sink_tensor,
            output,
            lse,
            sm_scale,
            total_tokens,
            _bucket_total_tokens(total_tokens),
            h_q,
            topk,
            num_blocks,
            block_size,
            s_q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            stride_kv_block,
            indices.stride(0),
            indices.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            HAS_TOPK_LENGTH=topk_length is not None,
            HAS_ATTN_SINK=attn_sink is not None,
        )

    if disable_buffer_ops:
        with triton.knobs.amd.scope():
            triton.knobs.amd.use_buffer_ops = False
            run_kernel()
    else:
        run_kernel()

    return output, lse


# Uses helper function to eliminate code duplication
# ============================================================================
@triton.autotune(
    configs=[
        # num_warps=4, num_stages=1 configs
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        # num_warps=8, num_stages=1 configs
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        # num_stages=2 configs (software pipelining)
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    ],
    key=["total_tokens_bucket", "h_q", "topk_main", "topk_extra"],
)
@triton.jit
def _fused_gather_attn_dsv4_dual_scope_kernel(
    Q,
    KV_Cache_Main,
    Indices_Main,
    TopkLength_Main,
    KV_Cache_Extra,
    Indices_Extra,
    TopkLength_Extra,
    AttnSink,
    Output,
    LSE,
    sm_scale,
    total_tokens,
    total_tokens_bucket,
    h_q,
    topk_main,
    num_blocks_main,
    block_size_main,
    topk_extra,
    num_blocks_extra,
    block_size_extra,
    s_q,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_block_main,
    stride_kv_block_extra,
    stride_idx_main_t,
    stride_idx_main_k,
    stride_idx_extra_t,
    stride_idx_extra_k,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_t,
    stride_lse_h,
    HAS_TOPK_LENGTH_MAIN: tl.constexpr,
    HAS_TOPK_LENGTH_EXTRA: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    OPTIMIZED fused gather+dequant+attention kernel for DSV4 with dual scope.

    This version uses a helper function (_process_kv_block_aggressive) to
    eliminate the ~200 lines of duplicated code between MAIN and EXTRA scope
    processing loops.

    The kernel processes:
    1. MAIN scope: topk_main tokens from KV_Cache_Main
    2. EXTRA scope: topk_extra tokens from KV_Cache_Extra

    Both scopes contribute to the same online softmax accumulator.
    """
    LOG2E: tl.constexpr = 1.4426950408889634
    D_NOPE: tl.constexpr = 448
    D_ROPE: tl.constexpr = 64
    TILE_SIZE: tl.constexpr = 64
    BYTES_PER_TOKEN_DATA: tl.constexpr = 576
    BYTES_PER_TOKEN_SCALE: tl.constexpr = 8

    # OPTIMIZED: Swapped grid - pid_h first for better cache locality
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    NEG_INF = float("-inf")

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    # Initialize accumulators
    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_4 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_5 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_6 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_7 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64)
    q_base = Q + pid_t_64 * stride_q_t_64

    batch_idx = pid_t // s_q
    offs_tile = tl.arange(0, TILE_SIZE)

    # Load Q tiles (shared by both scopes)
    q_0 = tl.load(
        q_base + offs_h[:, None] * stride_q_h + offs_tile[None, :] * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_1 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_2 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (2 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_3 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (3 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_4 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (4 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_5 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (5 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_6 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (6 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_7 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (7 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    # ========================================================================
    # Process MAIN scope
    # ========================================================================
    # Early-exit: pre-load topk_len and skip invalid blocks
    if HAS_TOPK_LENGTH_MAIN:
        topk_len = tl.load(TopkLength_Main + batch_idx)

    for n_start in range(0, topk_main, BLOCK_N):
        # Skip entire block if beyond valid topk range
        should_compute = not HAS_TOPK_LENGTH_MAIN or n_start < topk_len
        if should_compute:
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < topk_main

            idx_ptrs = (
                Indices_Main + pid_t * stride_idx_main_t + offs_n * stride_idx_main_k
            )
            indices = tl.load(idx_ptrs, mask=mask_n, other=-1)

            is_invalid = indices == -1
            if HAS_TOPK_LENGTH_MAIN:
                is_invalid = is_invalid | (offs_n >= topk_len)

            valid = mask_n & ~is_invalid
            indices_clamped = tl.maximum(indices, 0)

            block_idx = indices_clamped // block_size_main
            offset_in_block = indices_clamped % block_size_main

            block_idx_64 = block_idx.to(tl.int64)
            offset_in_block_64 = offset_in_block.to(tl.int64)

            stride_kv_block_main_64 = tl.cast(stride_kv_block_main, tl.int64)
            kv_block_base = KV_Cache_Main + block_idx_64 * stride_kv_block_main_64
            nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
            scale_base_offset = (
                block_size_main * BYTES_PER_TOKEN_DATA
                + offset_in_block_64 * BYTES_PER_TOKEN_SCALE
            )

            valid_2d = valid[:, None]

            # Use helper function for KV processing
            acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = (
                _process_kv_block_aggressive(
                    kv_block_base,
                    nope_rope_offset,
                    scale_base_offset,
                    valid,
                    valid_2d,
                    q_0,
                    q_1,
                    q_2,
                    q_3,
                    q_4,
                    q_5,
                    q_6,
                    q_7,
                    acc_0,
                    acc_1,
                    acc_2,
                    acc_3,
                    acc_4,
                    acc_5,
                    acc_6,
                    acc_7,
                    m_i,
                    l_i,
                    offs_tile,
                    sm_scale,
                    TILE_SIZE,
                    D_NOPE,
                    LOG2E,
                    BLOCK_H,
                    BLOCK_N,
                )
            )

    # ========================================================================
    # Process EXTRA scope
    # ========================================================================
    # Early-exit: pre-load topk_len and skip invalid blocks
    if HAS_TOPK_LENGTH_EXTRA:
        topk_len = tl.load(TopkLength_Extra + batch_idx)

    for n_start in range(0, topk_extra, BLOCK_N):
        # Skip entire block if beyond valid topk range
        should_compute = not HAS_TOPK_LENGTH_EXTRA or n_start < topk_len
        if should_compute:
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < topk_extra

            idx_ptrs = (
                Indices_Extra + pid_t * stride_idx_extra_t + offs_n * stride_idx_extra_k
            )
            indices = tl.load(idx_ptrs, mask=mask_n, other=-1)

            is_invalid = indices == -1
            if HAS_TOPK_LENGTH_EXTRA:
                is_invalid = is_invalid | (offs_n >= topk_len)

            valid = mask_n & ~is_invalid
            indices_clamped = tl.maximum(indices, 0)

            block_idx = indices_clamped // block_size_extra
            offset_in_block = indices_clamped % block_size_extra

            block_idx_64 = block_idx.to(tl.int64)
            offset_in_block_64 = offset_in_block.to(tl.int64)

            stride_kv_block_extra_64 = tl.cast(stride_kv_block_extra, tl.int64)
            kv_block_base = KV_Cache_Extra + block_idx_64 * stride_kv_block_extra_64
            nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
            scale_base_offset = (
                block_size_extra * BYTES_PER_TOKEN_DATA
                + offset_in_block_64 * BYTES_PER_TOKEN_SCALE
            )

            valid_2d = valid[:, None]

            # Use helper function for KV processing
            acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = (
                _process_kv_block_aggressive(
                    kv_block_base,
                    nope_rope_offset,
                    scale_base_offset,
                    valid,
                    valid_2d,
                    q_0,
                    q_1,
                    q_2,
                    q_3,
                    q_4,
                    q_5,
                    q_6,
                    q_7,
                    acc_0,
                    acc_1,
                    acc_2,
                    acc_3,
                    acc_4,
                    acc_5,
                    acc_6,
                    acc_7,
                    m_i,
                    l_i,
                    offs_tile,
                    sm_scale,
                    TILE_SIZE,
                    D_NOPE,
                    LOG2E,
                    BLOCK_H,
                    BLOCK_N,
                )
            )

    # ========================================================================
    # Finalize: compute LSE and output
    # ========================================================================
    lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
    is_lonely_q = l_i == 0.0

    # Compute output scale
    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)
        denominator = l_i + exp_attn_sink_minus_m
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

    # Apply output scaling and handle lonely queries
    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0 * output_scale[:, None])
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1 * output_scale[:, None])
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2 * output_scale[:, None])
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3 * output_scale[:, None])
    acc_4 = tl.where(is_lonely_q[:, None], 0.0, acc_4 * output_scale[:, None])
    acc_5 = tl.where(is_lonely_q[:, None], 0.0, acc_5 * output_scale[:, None])
    acc_6 = tl.where(is_lonely_q[:, None], 0.0, acc_6 * output_scale[:, None])
    acc_7 = tl.where(is_lonely_q[:, None], 0.0, acc_7 * output_scale[:, None])
    lse = tl.where(is_lonely_q, float("+inf"), lse)

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64

    # Optimized output stores with pre-computed row base pointers
    # Convert to bfloat16 first (batch conversion)
    o_0 = acc_0.to(tl.bfloat16)
    o_1 = acc_1.to(tl.bfloat16)
    o_2 = acc_2.to(tl.bfloat16)
    o_3 = acc_3.to(tl.bfloat16)
    o_4 = acc_4.to(tl.bfloat16)
    o_5 = acc_5.to(tl.bfloat16)
    o_6 = acc_6.to(tl.bfloat16)
    o_7 = acc_7.to(tl.bfloat16)

    # Pre-compute row base pointers (shared across all 8 stores)
    row_ptrs = o_base + offs_h[:, None] * stride_o_h

    # Store all 8 tiles with optimized pointer arithmetic
    tl.store(row_ptrs + offs_tile[None, :] * stride_o_d, o_0, mask=mask_h[:, None])
    tl.store(
        row_ptrs + (TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_1,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (2 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_2,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (3 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_3,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (4 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_4,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (5 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_5,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (6 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_6,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (7 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_7,
        mask=mask_h[:, None],
    )

    lse_ptrs = LSE + pid_t * stride_lse_t + offs_h * stride_lse_h
    tl.store(lse_ptrs, lse, mask=mask_h)



def _prune_splitk_configs(configs, named_args, **kwargs):
    """Prune BLOCK_H=16 configs for large batch sizes to avoid CU oversubscription.
    
    With h_q=128 and BLOCK_H=16, the grid has cdiv(128,16)=8 H-blocks.
    At bs=32 with split_k=2, this creates 8*32*2=512 blocks (200% CU),
    causing performance regression from oversubscription.
    
    For small batch sizes (bucket <= 8), BLOCK_H=16 provides better
    parallelism and is ~10% faster in CUDA graph replay.
    """
    total_tokens_bucket = named_args.get("total_tokens_bucket", 32)
    if total_tokens_bucket > 8:
        # Remove BLOCK_H=16 configs for large batch sizes
        pruned = [c for c in configs if c.kwargs.get("BLOCK_H", 32) > 16]
        if pruned:
            return pruned
    return configs


# ============================================================================
# Split-K Kernel for Dual Scope
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        # BLOCK_H=16 for better parallelism at small batch sizes
        # (pruned for large batch sizes by _prune_splitk_configs)
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_N": 256}, num_warps=4, num_stages=1),
    ],
    key=["total_tokens_bucket", "h_q", "topk_per_split"],
    prune_configs_by={"early_config_prune": _prune_splitk_configs},
)
@triton.jit
def _fused_gather_attn_dsv4_dual_scope_splitk_kernel(
    Q,
    KV_Cache_Main,
    Indices_Main,
    TopkLength_Main,
    KV_Cache_Extra,
    Indices_Extra,
    TopkLength_Extra,
    PartialOutput,
    PartialLSE,
    sm_scale,
    total_tokens,
    total_tokens_bucket,
    h_q,
    topk_main,
    num_blocks_main,
    block_size_main,
    topk_extra,
    num_blocks_extra,
    block_size_extra,
    s_q,
    topk_per_split,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_block_main,
    stride_kv_block_extra,
    stride_idx_main_t,
    stride_idx_main_k,
    stride_idx_extra_t,
    stride_idx_extra_k,
    stride_po_s,
    stride_po_t,
    stride_po_h,
    stride_po_d,
    stride_plse_s,
    stride_plse_t,
    stride_plse_h,
    HAS_TOPK_LENGTH_MAIN: tl.constexpr,
    HAS_TOPK_LENGTH_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Split-K fused gather+dequant+attention kernel for DSV4 with dual scope.

    This kernel processes a portion of the combined topk range (main + extra).
    Each split handles topk_per_split tokens from the combined range.
    """
    LOG2E: tl.constexpr = 1.4426950408889634
    D_NOPE: tl.constexpr = 448
    TILE_SIZE: tl.constexpr = 64
    BYTES_PER_TOKEN_DATA: tl.constexpr = 576
    BYTES_PER_TOKEN_SCALE: tl.constexpr = 8

    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)
    pid_t_64 = pid_t.to(tl.int64)

    NEG_INF = float("-inf")

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    # Calculate the range for this split
    total_topk = topk_main + topk_extra
    k_start = pid_k * topk_per_split
    k_end = tl.minimum(k_start + topk_per_split, total_topk)

    # Initialize accumulators
    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_4 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_5 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_6 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_7 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64)
    q_base = Q + pid_t_64 * stride_q_t_64

    batch_idx = pid_t // s_q
    offs_tile = tl.arange(0, TILE_SIZE)

    # Load Q tiles (shared by both scopes)
    q_row_base = q_base + offs_h[:, None] * stride_q_h
    q_0 = tl.load(
        q_row_base + offs_tile[None, :] * stride_q_d, mask=mask_h[:, None], other=0.0
    ).to(tl.bfloat16)
    q_1 = tl.load(
        q_row_base + (TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_2 = tl.load(
        q_row_base + (2 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_3 = tl.load(
        q_row_base + (3 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_4 = tl.load(
        q_row_base + (4 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_5 = tl.load(
        q_row_base + (5 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_6 = tl.load(
        q_row_base + (6 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_7 = tl.load(
        q_row_base + (7 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    stride_kv_block_main_64 = tl.cast(stride_kv_block_main, tl.int64)
    stride_kv_block_extra_64 = tl.cast(stride_kv_block_extra, tl.int64)

    # Process the combined range [k_start, k_end)
    # First, process MAIN scope portion (indices 0 to topk_main-1)
    main_start = k_start
    main_end = tl.minimum(k_end, topk_main)

    # Early-exit: pre-load topk_len and skip invalid blocks
    if HAS_TOPK_LENGTH_MAIN:
        topk_len = tl.load(TopkLength_Main + batch_idx)

    for n_start in range(main_start, main_end, BLOCK_N):
        # Skip entire block if beyond valid topk range
        should_compute = not HAS_TOPK_LENGTH_MAIN or n_start < topk_len
        if should_compute:
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < main_end

            idx_ptrs = (
                Indices_Main + pid_t * stride_idx_main_t + offs_n * stride_idx_main_k
            )
            indices = tl.load(idx_ptrs, mask=mask_n, other=-1)

            is_invalid = indices == -1
            if HAS_TOPK_LENGTH_MAIN:
                is_invalid = is_invalid | (offs_n >= topk_len)

            valid = mask_n & ~is_invalid
            indices_clamped = tl.maximum(indices, 0)

            block_idx = indices_clamped // block_size_main
            offset_in_block = indices_clamped % block_size_main

            block_idx_64 = block_idx.to(tl.int64)
            offset_in_block_64 = offset_in_block.to(tl.int64)

            kv_block_base = KV_Cache_Main + block_idx_64 * stride_kv_block_main_64
            nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
            scale_base_offset = (
                block_size_main * BYTES_PER_TOKEN_DATA
                + offset_in_block_64 * BYTES_PER_TOKEN_SCALE
            )

            valid_2d = valid[:, None]

            acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = (
                _process_kv_block_aggressive(
                    kv_block_base,
                    nope_rope_offset,
                    scale_base_offset,
                    valid,
                    valid_2d,
                    q_0,
                    q_1,
                    q_2,
                    q_3,
                    q_4,
                    q_5,
                    q_6,
                    q_7,
                    acc_0,
                    acc_1,
                    acc_2,
                    acc_3,
                    acc_4,
                    acc_5,
                    acc_6,
                    acc_7,
                    m_i,
                    l_i,
                    offs_tile,
                    sm_scale,
                    TILE_SIZE,
                    D_NOPE,
                    LOG2E,
                    BLOCK_H,
                    BLOCK_N,
                )
            )

    # Process EXTRA scope portion (indices topk_main to topk_main+topk_extra-1)
    extra_global_start = tl.maximum(k_start, topk_main)
    extra_global_end = k_end

    # Early-exit: pre-load topk_len and skip invalid blocks
    if HAS_TOPK_LENGTH_EXTRA:
        topk_len = tl.load(TopkLength_Extra + batch_idx)

    for n_global in range(extra_global_start, extra_global_end, BLOCK_N):
        # Skip entire block if beyond valid topk range
        should_compute = not HAS_TOPK_LENGTH_EXTRA or (n_global - topk_main) < topk_len
        if should_compute:
            offs_n_local = (n_global - topk_main) + tl.arange(0, BLOCK_N)
            offs_n_global = n_global + tl.arange(0, BLOCK_N)
            mask_n = offs_n_global < extra_global_end

            idx_ptrs = (
                Indices_Extra
                + pid_t * stride_idx_extra_t
                + offs_n_local * stride_idx_extra_k
            )
            indices = tl.load(idx_ptrs, mask=mask_n, other=-1)

            is_invalid = indices == -1
            if HAS_TOPK_LENGTH_EXTRA:
                is_invalid = is_invalid | (offs_n_local >= topk_len)

            valid = mask_n & ~is_invalid
            indices_clamped = tl.maximum(indices, 0)

            block_idx = indices_clamped // block_size_extra
            offset_in_block = indices_clamped % block_size_extra

            block_idx_64 = block_idx.to(tl.int64)
            offset_in_block_64 = offset_in_block.to(tl.int64)

            kv_block_base = KV_Cache_Extra + block_idx_64 * stride_kv_block_extra_64
            nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
            scale_base_offset = (
                block_size_extra * BYTES_PER_TOKEN_DATA
                + offset_in_block_64 * BYTES_PER_TOKEN_SCALE
            )

            valid_2d = valid[:, None]

            acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = (
                _process_kv_block_aggressive(
                    kv_block_base,
                    nope_rope_offset,
                    scale_base_offset,
                    valid,
                    valid_2d,
                    q_0,
                    q_1,
                    q_2,
                    q_3,
                    q_4,
                    q_5,
                    q_6,
                    q_7,
                    acc_0,
                    acc_1,
                    acc_2,
                    acc_3,
                    acc_4,
                    acc_5,
                    acc_6,
                    acc_7,
                    m_i,
                    l_i,
                    offs_tile,
                    sm_scale,
                    TILE_SIZE,
                    D_NOPE,
                    LOG2E,
                    BLOCK_H,
                    BLOCK_N,
                )
            )

    # Finalize: compute partial LSE and store partial output
    lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
    is_lonely_q = l_i == 0.0

    output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0 * output_scale[:, None])
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1 * output_scale[:, None])
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2 * output_scale[:, None])
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3 * output_scale[:, None])
    acc_4 = tl.where(is_lonely_q[:, None], 0.0, acc_4 * output_scale[:, None])
    acc_5 = tl.where(is_lonely_q[:, None], 0.0, acc_5 * output_scale[:, None])
    acc_6 = tl.where(is_lonely_q[:, None], 0.0, acc_6 * output_scale[:, None])
    acc_7 = tl.where(is_lonely_q[:, None], 0.0, acc_7 * output_scale[:, None])
    lse = tl.where(is_lonely_q, float("+inf"), lse)

    # Store partial output
    stride_po_s_64 = tl.cast(stride_po_s, tl.int64)
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64)
    po_base = PartialOutput + pid_k * stride_po_s_64 + pid_t_64 * stride_po_t_64

    # Store partial output as float32 for better precision in combine kernel
    row_ptrs = po_base + offs_h[:, None] * stride_po_h

    tl.store(row_ptrs + offs_tile[None, :] * stride_po_d, acc_0, mask=mask_h[:, None])
    tl.store(
        row_ptrs + (TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_1,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (2 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_2,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (3 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_3,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (4 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_4,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (5 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_5,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (6 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_6,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (7 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_7,
        mask=mask_h[:, None],
    )

    # Store partial LSE
    stride_plse_s_64 = tl.cast(stride_plse_s, tl.int64)
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64)
    lse_ptrs = (
        PartialLSE
        + pid_k * stride_plse_s_64
        + pid_t_64 * stride_plse_t_64
        + offs_h * stride_plse_h
    )
    tl.store(lse_ptrs, lse, mask=mask_h)


def fused_gather_attn_decode_dsv4_dual_scope(
    q: torch.Tensor,
    kv_cache_main: torch.Tensor,
    indices_main: torch.Tensor,
    block_size_main: int,
    kv_cache_extra: torch.Tensor,
    indices_extra: torch.Tensor,
    block_size_extra: int,
    sm_scale: float,
    topk_length_main: Optional[torch.Tensor] = None,
    topk_length_extra: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    s_q: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused gather+dequant+attention for DSV4 with dual scope (main + extra).
    Uses Split-K optimization for large total_topk (>= SPLITK_TOPK_THRESHOLD).

    Args:
        q: Query tensor [total_tokens, h_q, d_qk]
        kv_cache_main: Quantized main KV cache
        indices_main: Main KV indices [total_tokens, topk_main]
        block_size_main: Block size for main KV cache
        kv_cache_extra: Quantized extra KV cache
        indices_extra: Extra KV indices [total_tokens, topk_extra]
        block_size_extra: Block size for extra KV cache
        sm_scale: Softmax scale
        topk_length_main: Optional per-batch topk length for main [b]
        topk_length_extra: Optional per-batch topk length for extra [b]
        attn_sink: Optional attention sink values [h_q]
        s_q: Sequence length per batch

    Returns:
        output: Attention output [total_tokens, h_q, d_v]
        lse: Log-sum-exp values [total_tokens, h_q]
    """
    total_tokens, h_q, d_qk = q.shape
    topk_main = indices_main.shape[1]
    topk_extra = indices_extra.shape[1]
    total_topk = topk_main + topk_extra
    d_v = DSV4_D_V
    device = q.device

    # Prepare main KV cache
    kv_uint8_main = kv_cache_main.view(torch.uint8)
    num_blocks_main = kv_cache_main.shape[0]
    stride_kv_block_main = kv_uint8_main.stride(0)
    kv_flat_main = kv_uint8_main.reshape(num_blocks_main, -1)

    # Prepare extra KV cache
    kv_uint8_extra = kv_cache_extra.view(torch.uint8)
    num_blocks_extra = kv_cache_extra.shape[0]
    stride_kv_block_extra = kv_uint8_extra.stride(0)
    kv_flat_extra = kv_uint8_extra.reshape(num_blocks_extra, -1)

    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        q = q.to(torch.bfloat16).contiguous()

    if not indices_main.is_contiguous():
        indices_main = indices_main.contiguous()
    if not indices_extra.is_contiguous():
        indices_extra = indices_extra.contiguous()

    kv_cache_size_main = stride_kv_block_main * num_blocks_main
    kv_cache_size_extra = stride_kv_block_extra * num_blocks_extra
    disable_buffer_ops = (
        kv_cache_size_main > BUFFER_OPS_DISABLE_THRESHOLD
        or kv_cache_size_extra > BUFFER_OPS_DISABLE_THRESHOLD
    )

    # Use Split-K for dual scope in these cases:
    # 1. Small batch sizes with h_q=128 or large topk to increase GPU parallelism
    # 2. Large topk (>= 2048) with medium/large batch sizes
    # 3. NEW: h_q=64 + large topk (>=1024) + medium batch sizes (~21% improvement)
    SPLITK_DUAL_SCOPE_TOPK_THRESHOLD = 2048
    # For small bs, only use splitk when h_q=128 or total_topk >= 1024
    use_splitk_for_small_bs = total_tokens <= 8 and (h_q >= 128 or total_topk >= 1024)
    # NEW: For h_q=64 with large topk, splitk is beneficial for medium batch sizes
    # Only for tokens <= 128 based on benchmarking (bs=64 shows 13% improvement)
    use_splitk_for_h64_large_topk = (
        h_q <= 64 and total_topk >= 1024 and total_tokens > 8 and total_tokens <= 128
    )
    use_splitk_for_large_topk = (
        total_tokens > 64 and total_topk >= SPLITK_DUAL_SCOPE_TOPK_THRESHOLD
    )
    # For h_q > 64 (e.g. h_q=128), the non-splitk grid has very few blocks
    # in the H dimension, leading to low GPU utilization at medium batch sizes.
    use_splitk_for_large_hq = (
        h_q > 64 and total_tokens > 8 and total_topk >= 256
    )
    if (
        use_splitk_for_small_bs
        or use_splitk_for_h64_large_topk
        or use_splitk_for_large_topk
        or use_splitk_for_large_hq
    ):
        # Select split_k based on workload and total_topk.
        # CUDA graph replay benchmarks show optimal split_k depends on both:
        #   - High topk (>=512, c4 layers): more splits needed to parallelize
        #   - Low topk (<512, c128 layers): fewer splits, less combine overhead
        if total_tokens <= 8:
            if total_topk >= 512 and total_tokens <= 4:
                # High topk + very small bs: split_k=8 is 8-33% faster than sk=4
                split_k = 8
            else:
                # split_k=4 gives 2x more blocks than split_k=2
                split_k = 4
        elif use_splitk_for_large_hq:
            # For h_q > 64 with bs > 8:
            if total_topk >= 512:
                # High topk: split_k=4 for all medium/large bs
                split_k = 4
            else:
                # Low topk: split_k=2 is sufficient
                split_k = 2
        elif use_splitk_for_h64_large_topk:
            # For h_q=64 + large topk + medium bs, split_k=2 is optimal
            split_k = 2
        else:
            split_k = _select_split_k(total_topk, h_q, total_tokens)
        topk_per_split = (total_topk + split_k - 1) // split_k

        partial_output = torch.empty(
            split_k, total_tokens, h_q, d_v, dtype=torch.float32, device=device
        )
        partial_lse = torch.empty(
            split_k, total_tokens, h_q, dtype=torch.float32, device=device
        )
        output = torch.empty(
            total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device
        )
        lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

        topk_length_main_tensor = (
            topk_length_main if topk_length_main is not None else lse[:1, 0]
        )
        topk_length_extra_tensor = (
            topk_length_extra if topk_length_extra is not None else lse[:1, 0]
        )
        attn_sink_tensor = attn_sink if attn_sink is not None else lse[0, :]

        grid_splitk = lambda meta: (
            triton.cdiv(h_q, meta["BLOCK_H"]),
            total_tokens,
            split_k,
        )

        def run_splitk_kernel():
            _fused_gather_attn_dsv4_dual_scope_splitk_kernel[grid_splitk](
                q,
                kv_flat_main,
                indices_main,
                topk_length_main_tensor,
                kv_flat_extra,
                indices_extra,
                topk_length_extra_tensor,
                partial_output,
                partial_lse,
                sm_scale,
                total_tokens,
                _bucket_total_tokens(total_tokens),
                h_q,
                topk_main,
                num_blocks_main,
                block_size_main,
                topk_extra,
                num_blocks_extra,
                block_size_extra,
                s_q,
                topk_per_split,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                stride_kv_block_main,
                stride_kv_block_extra,
                indices_main.stride(0),
                indices_main.stride(1),
                indices_extra.stride(0),
                indices_extra.stride(1),
                partial_output.stride(0),
                partial_output.stride(1),
                partial_output.stride(2),
                partial_output.stride(3),
                partial_lse.stride(0),
                partial_lse.stride(1),
                partial_lse.stride(2),
                HAS_TOPK_LENGTH_MAIN=topk_length_main is not None,
                HAS_TOPK_LENGTH_EXTRA=topk_length_extra is not None,
            )

        if disable_buffer_ops:
            with triton.knobs.amd.scope():
                triton.knobs.amd.use_buffer_ops = False
                run_splitk_kernel()
        else:
            run_splitk_kernel()

        # Use appropriate combine kernel based on split_k
        if split_k == 8:
            grid_combine = lambda meta: (
                total_tokens,
                triton.cdiv(h_q, meta["BLOCK_H"]),
            )
            _combine_splitk_kernel_8_optimized[grid_combine](
                partial_output,
                partial_lse,
                attn_sink_tensor,
                output,
                lse,
                total_tokens,
                _bucket_total_tokens(total_tokens),
                h_q,
                d_v,
                partial_output.stride(0),
                partial_output.stride(1),
                partial_output.stride(2),
                partial_output.stride(3),
                partial_lse.stride(0),
                partial_lse.stride(1),
                partial_lse.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                lse.stride(0),
                lse.stride(1),
                HAS_ATTN_SINK=attn_sink is not None,
            )
        else:
            BLOCK_H_COMBINE = 16
            BLOCK_D_COMBINE = 128
            grid_combine = (total_tokens, triton.cdiv(h_q, BLOCK_H_COMBINE))

            if split_k == 2:
                combine_kernel = _combine_splitk_kernel_2
            elif split_k == 4:
                combine_kernel = _combine_splitk_kernel
            else:
                raise ValueError(f"Unsupported split_k: {split_k}")

            combine_kernel[grid_combine](
                partial_output,
                partial_lse,
                attn_sink_tensor,
                output,
                lse,
                total_tokens,
                _bucket_total_tokens(total_tokens),
                h_q,
                d_v,
                partial_output.stride(0),
                partial_output.stride(1),
                partial_output.stride(2),
                partial_output.stride(3),
                partial_lse.stride(0),
                partial_lse.stride(1),
                partial_lse.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                lse.stride(0),
                lse.stride(1),
                HAS_ATTN_SINK=attn_sink is not None,
                BLOCK_H=BLOCK_H_COMBINE,
                BLOCK_D=BLOCK_D_COMBINE,
                num_warps=4,
                num_stages=1,
            )

        return output, lse

    # Use original kernel for smaller total_topk
    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

    topk_length_main_tensor = (
        topk_length_main if topk_length_main is not None else lse[:1, 0]
    )
    topk_length_extra_tensor = (
        topk_length_extra if topk_length_extra is not None else lse[:1, 0]
    )
    attn_sink_tensor = attn_sink if attn_sink is not None else lse[0, :]

    grid = lambda meta: (triton.cdiv(h_q, meta["BLOCK_H"]), total_tokens)

    def run_kernel():
        _fused_gather_attn_dsv4_dual_scope_kernel[grid](
            q,
            kv_flat_main,
            indices_main,
            topk_length_main_tensor,
            kv_flat_extra,
            indices_extra,
            topk_length_extra_tensor,
            attn_sink_tensor,
            output,
            lse,
            sm_scale,
            total_tokens,
            _bucket_total_tokens(total_tokens),
            h_q,
            topk_main,
            num_blocks_main,
            block_size_main,
            topk_extra,
            num_blocks_extra,
            block_size_extra,
            s_q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            stride_kv_block_main,
            stride_kv_block_extra,
            indices_main.stride(0),
            indices_main.stride(1),
            indices_extra.stride(0),
            indices_extra.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            HAS_TOPK_LENGTH_MAIN=topk_length_main is not None,
            HAS_TOPK_LENGTH_EXTRA=topk_length_extra is not None,
            HAS_ATTN_SINK=attn_sink is not None,
        )

    if disable_buffer_ops:
        with triton.knobs.amd.scope():
            triton.knobs.amd.use_buffer_ops = False
            run_kernel()
    else:
        run_kernel()

    return output, lse


# ============================================================================
# Split-K Optimization for Large TopK (>= 8192)
# ============================================================================
SPLITK_TOPK_THRESHOLD = 8192
SPLITK_DEFAULT = 4


@triton.autotune(
    configs=[
        # Tiny BLOCK_N=8 for minimal scattered access
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 8}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 8}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 8}, num_warps=8, num_stages=1),
        # Very small BLOCK_N=16
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 16}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 16}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 16}, num_warps=8, num_stages=1),
        # Small BLOCK_N=32
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 32}, num_warps=8, num_stages=1),
        # Medium BLOCK_N=64
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_N": 64}, num_warps=8, num_stages=1),
    ],
    key=["total_tokens_bucket", "h_q", "topk_per_split"],
)
@triton.jit
def _fused_gather_attn_dsv4_splitk_kernel(
    Q,
    KV_Cache,
    Indices,
    TopkLength,
    PartialOutput,
    PartialLSE,
    sm_scale,
    total_tokens,
    total_tokens_bucket,
    h_q,
    topk,
    num_blocks,
    block_size,
    s_q,
    topk_per_split,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_block,
    stride_idx_t,
    stride_idx_k,
    stride_po_s,
    stride_po_t,
    stride_po_h,
    stride_po_d,
    stride_plse_s,
    stride_plse_t,
    stride_plse_h,
    HAS_TOPK_LENGTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Split-K fused gather+dequant+attention kernel for DSV4."""
    LOG2E: tl.constexpr = 1.4426950408889634
    D_NOPE: tl.constexpr = 448
    TILE_SIZE: tl.constexpr = 64
    BYTES_PER_TOKEN_DATA: tl.constexpr = 576
    BYTES_PER_TOKEN_SCALE: tl.constexpr = 8

    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)
    pid_t_64 = pid_t.to(tl.int64)

    NEG_INF = float("-inf")

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    k_start = pid_k * topk_per_split
    k_end = tl.minimum(k_start + topk_per_split, topk)

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_4 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_5 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_6 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_7 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64)
    q_base = Q + pid_t_64 * stride_q_t_64

    batch_idx = pid_t // s_q
    offs_tile = tl.arange(0, TILE_SIZE)

    q_row_base = q_base + offs_h[:, None] * stride_q_h
    q_0 = tl.load(
        q_row_base + offs_tile[None, :] * stride_q_d, mask=mask_h[:, None], other=0.0
    ).to(tl.bfloat16)
    q_1 = tl.load(
        q_row_base + (TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_2 = tl.load(
        q_row_base + (2 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_3 = tl.load(
        q_row_base + (3 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_4 = tl.load(
        q_row_base + (4 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_5 = tl.load(
        q_row_base + (5 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_6 = tl.load(
        q_row_base + (6 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_7 = tl.load(
        q_row_base + (7 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    stride_kv_block_64 = tl.cast(stride_kv_block, tl.int64)

    # Early-exit: pre-load topk_len and skip invalid blocks
    if HAS_TOPK_LENGTH:
        topk_len = tl.load(TopkLength + batch_idx)

    for n_start in range(k_start, k_end, BLOCK_N):
        # Skip entire block if beyond valid topk range
        should_compute = not HAS_TOPK_LENGTH or n_start < topk_len
        if should_compute:
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < k_end

            idx_ptrs = Indices + pid_t * stride_idx_t + offs_n * stride_idx_k
            indices = tl.load(idx_ptrs, mask=mask_n, other=-1)

            is_invalid = indices == -1
            if HAS_TOPK_LENGTH:
                is_invalid = is_invalid | (offs_n >= topk_len)

            valid = mask_n & ~is_invalid
            indices_clamped = tl.maximum(indices, 0)

            block_idx = indices_clamped // block_size
            offset_in_block = indices_clamped % block_size

            block_idx_64 = block_idx.to(tl.int64)
            offset_in_block_64 = offset_in_block.to(tl.int64)

            kv_block_base = KV_Cache + block_idx_64 * stride_kv_block_64
            nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
            scale_base_offset = (
                block_size * BYTES_PER_TOKEN_DATA
                + offset_in_block_64 * BYTES_PER_TOKEN_SCALE
            )

            valid_2d = valid[:, None]

            # Use helper function for KV processing
            acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = (
                _process_kv_block_aggressive(
                    kv_block_base,
                    nope_rope_offset,
                    scale_base_offset,
                    valid,
                    valid_2d,
                    q_0,
                    q_1,
                    q_2,
                    q_3,
                    q_4,
                    q_5,
                    q_6,
                    q_7,
                    acc_0,
                    acc_1,
                    acc_2,
                    acc_3,
                    acc_4,
                    acc_5,
                    acc_6,
                    acc_7,
                    m_i,
                    l_i,
                    offs_tile,
                    sm_scale,
                    TILE_SIZE,
                    D_NOPE,
                    LOG2E,
                    BLOCK_H,
                    BLOCK_N,
                )
            )

    lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
    is_lonely_q = l_i == 0.0

    output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)
    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0 * output_scale[:, None])
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1 * output_scale[:, None])
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2 * output_scale[:, None])
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3 * output_scale[:, None])
    acc_4 = tl.where(is_lonely_q[:, None], 0.0, acc_4 * output_scale[:, None])
    acc_5 = tl.where(is_lonely_q[:, None], 0.0, acc_5 * output_scale[:, None])
    acc_6 = tl.where(is_lonely_q[:, None], 0.0, acc_6 * output_scale[:, None])
    acc_7 = tl.where(is_lonely_q[:, None], 0.0, acc_7 * output_scale[:, None])
    lse = tl.where(is_lonely_q, float("+inf"), lse)

    stride_po_s_64 = tl.cast(stride_po_s, tl.int64)
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64)
    po_base = PartialOutput + pid_k * stride_po_s_64 + pid_t_64 * stride_po_t_64
    row_ptrs = po_base + offs_h[:, None] * stride_po_h

    # Store partial output as float32 for better precision in combine kernel
    tl.store(row_ptrs + offs_tile[None, :] * stride_po_d, acc_0, mask=mask_h[:, None])
    tl.store(
        row_ptrs + (TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_1,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (2 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_2,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (3 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_3,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (4 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_4,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (5 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_5,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (6 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_6,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (7 * TILE_SIZE + offs_tile[None, :]) * stride_po_d,
        acc_7,
        mask=mask_h[:, None],
    )

    stride_plse_s_64 = tl.cast(stride_plse_s, tl.int64)
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64)
    plse_ptrs = (
        PartialLSE
        + pid_k * stride_plse_s_64
        + pid_t_64 * stride_plse_t_64
        + offs_h * stride_plse_h
    )
    tl.store(plse_ptrs, lse, mask=mask_h)


@triton.jit
def _combine_splitk_kernel(
    PartialOutput,
    PartialLSE,
    AttnSink,
    Output,
    LSE,
    total_tokens,
    total_tokens_bucket,
    h_q,
    d_v,
    stride_po_s,
    stride_po_t,
    stride_po_h,
    stride_po_d,
    stride_plse_s,
    stride_plse_t,
    stride_plse_h,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_t,
    stride_lse_h,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Combine partial results from split-K kernel (SPLIT_K=4)."""
    LOG2E: tl.constexpr = 1.4426950408889634
    NEG_INF = float("-inf")
    POS_INF = float("+inf")
    INF_THRESHOLD = 1e30

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    offs_d = tl.arange(0, BLOCK_D)

    stride_plse_s_64 = tl.cast(stride_plse_s, tl.int64)
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64)

    lse_0 = tl.load(
        PartialLSE
        + 0 * stride_plse_s_64
        + pid_t_64 * stride_plse_t_64
        + offs_h * stride_plse_h,
        mask=mask_h,
        other=POS_INF,
    )
    lse_1 = tl.load(
        PartialLSE
        + 1 * stride_plse_s_64
        + pid_t_64 * stride_plse_t_64
        + offs_h * stride_plse_h,
        mask=mask_h,
        other=POS_INF,
    )
    lse_2 = tl.load(
        PartialLSE
        + 2 * stride_plse_s_64
        + pid_t_64 * stride_plse_t_64
        + offs_h * stride_plse_h,
        mask=mask_h,
        other=POS_INF,
    )
    lse_3 = tl.load(
        PartialLSE
        + 3 * stride_plse_s_64
        + pid_t_64 * stride_plse_t_64
        + offs_h * stride_plse_h,
        mask=mask_h,
        other=POS_INF,
    )

    lse_0_valid = tl.abs(lse_0) < INF_THRESHOLD
    lse_1_valid = tl.abs(lse_1) < INF_THRESHOLD
    lse_2_valid = tl.abs(lse_2) < INF_THRESHOLD
    lse_3_valid = tl.abs(lse_3) < INF_THRESHOLD

    lse_0_safe = tl.where(lse_0_valid, lse_0, NEG_INF)
    lse_1_safe = tl.where(lse_1_valid, lse_1, NEG_INF)
    lse_2_safe = tl.where(lse_2_valid, lse_2, NEG_INF)
    lse_3_safe = tl.where(lse_3_valid, lse_3, NEG_INF)

    max_lse = tl.maximum(
        tl.maximum(lse_0_safe, lse_1_safe), tl.maximum(lse_2_safe, lse_3_safe)
    )

    exp_0 = tl.where(lse_0_valid, tl.math.exp2((lse_0_safe - max_lse) * LOG2E), 0.0)
    exp_1 = tl.where(lse_1_valid, tl.math.exp2((lse_1_safe - max_lse) * LOG2E), 0.0)
    exp_2 = tl.where(lse_2_valid, tl.math.exp2((lse_2_safe - max_lse) * LOG2E), 0.0)
    exp_3 = tl.where(lse_3_valid, tl.math.exp2((lse_3_safe - max_lse) * LOG2E), 0.0)

    sum_exp = exp_0 + exp_1 + exp_2 + exp_3
    all_invalid = sum_exp == 0.0
    sum_exp_safe = tl.where(all_invalid, 1.0, sum_exp)

    combined_lse = max_lse + tl.math.log2(sum_exp_safe) / LOG2E
    combined_lse = tl.where(all_invalid, POS_INF, combined_lse)

    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        is_lonely = combined_lse > INF_THRESHOLD
        lse_safe_for_sink = tl.where(is_lonely, 0.0, combined_lse)
        diff = attn_sink_vals - lse_safe_for_sink
        diff_clamped = tl.minimum(tl.maximum(diff, -100.0), 100.0)
        exp_diff = tl.math.exp2(diff_clamped * LOG2E)
        exp_diff = tl.where(is_lonely, 0.0, exp_diff)
        denominator = 1.0 + exp_diff
        sink_scale = 1.0 / denominator
        sink_scale = tl.where(is_lonely, 1.0, sink_scale)

        scale_0 = (exp_0 / sum_exp_safe) * sink_scale
        scale_1 = (exp_1 / sum_exp_safe) * sink_scale
        scale_2 = (exp_2 / sum_exp_safe) * sink_scale
        scale_3 = (exp_3 / sum_exp_safe) * sink_scale
    else:
        scale_0 = exp_0 / sum_exp_safe
        scale_1 = exp_1 / sum_exp_safe
        scale_2 = exp_2 / sum_exp_safe
        scale_3 = exp_3 / sum_exp_safe

    scale_0 = tl.where(all_invalid, 0.0, scale_0)
    scale_1 = tl.where(all_invalid, 0.0, scale_1)
    scale_2 = tl.where(all_invalid, 0.0, scale_2)
    scale_3 = tl.where(all_invalid, 0.0, scale_3)

    stride_po_s_64 = tl.cast(stride_po_s, tl.int64)
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64)

    po_base_0 = (
        PartialOutput
        + 0 * stride_po_s_64
        + pid_t_64 * stride_po_t_64
        + offs_h[:, None] * stride_po_h
    )
    po_base_1 = (
        PartialOutput
        + 1 * stride_po_s_64
        + pid_t_64 * stride_po_t_64
        + offs_h[:, None] * stride_po_h
    )
    po_base_2 = (
        PartialOutput
        + 2 * stride_po_s_64
        + pid_t_64 * stride_po_t_64
        + offs_h[:, None] * stride_po_h
    )
    po_base_3 = (
        PartialOutput
        + 3 * stride_po_s_64
        + pid_t_64 * stride_po_t_64
        + offs_h[:, None] * stride_po_h
    )

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64 + offs_h[:, None] * stride_o_h

    for d_idx in range(4):
        d_offs = d_idx * BLOCK_D + offs_d[None, :]
        po_0 = tl.load(
            po_base_0 + d_offs * stride_po_d, mask=mask_h[:, None], other=0.0
        )
        po_1 = tl.load(
            po_base_1 + d_offs * stride_po_d, mask=mask_h[:, None], other=0.0
        )
        po_2 = tl.load(
            po_base_2 + d_offs * stride_po_d, mask=mask_h[:, None], other=0.0
        )
        po_3 = tl.load(
            po_base_3 + d_offs * stride_po_d, mask=mask_h[:, None], other=0.0
        )
        combined = (
            scale_0[:, None] * po_0
            + scale_1[:, None] * po_1
            + scale_2[:, None] * po_2
            + scale_3[:, None] * po_3
        )
        tl.store(
            o_base + d_offs * stride_o_d, combined.to(tl.bfloat16), mask=mask_h[:, None]
        )

    stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)
    lse_ptrs = LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h
    tl.store(lse_ptrs, combined_lse, mask=mask_h)


@triton.autotune(
    configs=[
        # Reduced from 15 to 5 configs. This is a simple reduce kernel
        # (weighted sum of 8 partial results), so performance is not
        # sensitive to tile shape. BLOCK_D=512 covers d_v=512 in one
        # pass; BLOCK_D=256 as fallback. BLOCK_H=16/32/64 covers the
        # parallelism range for small batch sizes (split_k=8 → bs<=4).
        triton.Config({"BLOCK_H": 16, "BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 64, "BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 16, "BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 32, "BLOCK_D": 256}, num_warps=4, num_stages=1),
    ],
    key=["total_tokens_bucket", "h_q", "d_v"],
)
@triton.jit
def _combine_splitk_kernel_8_optimized(
    PartialOutput,
    PartialLSE,
    AttnSink,
    Output,
    LSE,
    total_tokens,
    total_tokens_bucket,
    h_q,
    d_v,
    stride_po_s,
    stride_po_t,
    stride_po_h,
    stride_po_d,
    stride_plse_s,
    stride_plse_t,
    stride_plse_h,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_t,
    stride_lse_h,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Optimized combine kernel for split-K=8 with autotuning for BLOCK_H."""
    LOG2E: tl.constexpr = 1.4426950408889634
    NEG_INF = float("-inf")
    POS_INF = float("+inf")
    INF_THRESHOLD = 1e30

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    offs_d = tl.arange(0, BLOCK_D)

    stride_plse_s_64 = tl.cast(stride_plse_s, tl.int64)
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64)

    # Load all 8 LSE values
    lse_base = PartialLSE + pid_t_64 * stride_plse_t_64 + offs_h * stride_plse_h
    lse_0 = tl.load(lse_base + 0 * stride_plse_s_64, mask=mask_h, other=POS_INF)
    lse_1 = tl.load(lse_base + 1 * stride_plse_s_64, mask=mask_h, other=POS_INF)
    lse_2 = tl.load(lse_base + 2 * stride_plse_s_64, mask=mask_h, other=POS_INF)
    lse_3 = tl.load(lse_base + 3 * stride_plse_s_64, mask=mask_h, other=POS_INF)
    lse_4 = tl.load(lse_base + 4 * stride_plse_s_64, mask=mask_h, other=POS_INF)
    lse_5 = tl.load(lse_base + 5 * stride_plse_s_64, mask=mask_h, other=POS_INF)
    lse_6 = tl.load(lse_base + 6 * stride_plse_s_64, mask=mask_h, other=POS_INF)
    lse_7 = tl.load(lse_base + 7 * stride_plse_s_64, mask=mask_h, other=POS_INF)

    lse_0_valid = tl.abs(lse_0) < INF_THRESHOLD
    lse_1_valid = tl.abs(lse_1) < INF_THRESHOLD
    lse_2_valid = tl.abs(lse_2) < INF_THRESHOLD
    lse_3_valid = tl.abs(lse_3) < INF_THRESHOLD
    lse_4_valid = tl.abs(lse_4) < INF_THRESHOLD
    lse_5_valid = tl.abs(lse_5) < INF_THRESHOLD
    lse_6_valid = tl.abs(lse_6) < INF_THRESHOLD
    lse_7_valid = tl.abs(lse_7) < INF_THRESHOLD

    lse_0_safe = tl.where(lse_0_valid, lse_0, NEG_INF)
    lse_1_safe = tl.where(lse_1_valid, lse_1, NEG_INF)
    lse_2_safe = tl.where(lse_2_valid, lse_2, NEG_INF)
    lse_3_safe = tl.where(lse_3_valid, lse_3, NEG_INF)
    lse_4_safe = tl.where(lse_4_valid, lse_4, NEG_INF)
    lse_5_safe = tl.where(lse_5_valid, lse_5, NEG_INF)
    lse_6_safe = tl.where(lse_6_valid, lse_6, NEG_INF)
    lse_7_safe = tl.where(lse_7_valid, lse_7, NEG_INF)

    max_lse = tl.maximum(
        tl.maximum(
            tl.maximum(lse_0_safe, lse_1_safe), tl.maximum(lse_2_safe, lse_3_safe)
        ),
        tl.maximum(
            tl.maximum(lse_4_safe, lse_5_safe), tl.maximum(lse_6_safe, lse_7_safe)
        ),
    )

    exp_0 = tl.where(lse_0_valid, tl.math.exp2((lse_0_safe - max_lse) * LOG2E), 0.0)
    exp_1 = tl.where(lse_1_valid, tl.math.exp2((lse_1_safe - max_lse) * LOG2E), 0.0)
    exp_2 = tl.where(lse_2_valid, tl.math.exp2((lse_2_safe - max_lse) * LOG2E), 0.0)
    exp_3 = tl.where(lse_3_valid, tl.math.exp2((lse_3_safe - max_lse) * LOG2E), 0.0)
    exp_4 = tl.where(lse_4_valid, tl.math.exp2((lse_4_safe - max_lse) * LOG2E), 0.0)
    exp_5 = tl.where(lse_5_valid, tl.math.exp2((lse_5_safe - max_lse) * LOG2E), 0.0)
    exp_6 = tl.where(lse_6_valid, tl.math.exp2((lse_6_safe - max_lse) * LOG2E), 0.0)
    exp_7 = tl.where(lse_7_valid, tl.math.exp2((lse_7_safe - max_lse) * LOG2E), 0.0)

    sum_exp = exp_0 + exp_1 + exp_2 + exp_3 + exp_4 + exp_5 + exp_6 + exp_7
    all_invalid = sum_exp == 0.0
    sum_exp_safe = tl.where(all_invalid, 1.0, sum_exp)

    combined_lse = max_lse + tl.math.log2(sum_exp_safe) / LOG2E
    combined_lse = tl.where(all_invalid, POS_INF, combined_lse)

    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        is_lonely = combined_lse > INF_THRESHOLD
        lse_safe_for_sink = tl.where(is_lonely, 0.0, combined_lse)
        diff = attn_sink_vals - lse_safe_for_sink
        diff_clamped = tl.minimum(tl.maximum(diff, -100.0), 100.0)
        exp_diff = tl.math.exp2(diff_clamped * LOG2E)
        exp_diff = tl.where(is_lonely, 0.0, exp_diff)
        denominator = 1.0 + exp_diff
        sink_scale = 1.0 / denominator
        sink_scale = tl.where(is_lonely, 1.0, sink_scale)

        scale_0 = (exp_0 / sum_exp_safe) * sink_scale
        scale_1 = (exp_1 / sum_exp_safe) * sink_scale
        scale_2 = (exp_2 / sum_exp_safe) * sink_scale
        scale_3 = (exp_3 / sum_exp_safe) * sink_scale
        scale_4 = (exp_4 / sum_exp_safe) * sink_scale
        scale_5 = (exp_5 / sum_exp_safe) * sink_scale
        scale_6 = (exp_6 / sum_exp_safe) * sink_scale
        scale_7 = (exp_7 / sum_exp_safe) * sink_scale
    else:
        scale_0 = exp_0 / sum_exp_safe
        scale_1 = exp_1 / sum_exp_safe
        scale_2 = exp_2 / sum_exp_safe
        scale_3 = exp_3 / sum_exp_safe
        scale_4 = exp_4 / sum_exp_safe
        scale_5 = exp_5 / sum_exp_safe
        scale_6 = exp_6 / sum_exp_safe
        scale_7 = exp_7 / sum_exp_safe

    scale_0 = tl.where(all_invalid, 0.0, scale_0)
    scale_1 = tl.where(all_invalid, 0.0, scale_1)
    scale_2 = tl.where(all_invalid, 0.0, scale_2)
    scale_3 = tl.where(all_invalid, 0.0, scale_3)
    scale_4 = tl.where(all_invalid, 0.0, scale_4)
    scale_5 = tl.where(all_invalid, 0.0, scale_5)
    scale_6 = tl.where(all_invalid, 0.0, scale_6)
    scale_7 = tl.where(all_invalid, 0.0, scale_7)

    stride_po_s_64 = tl.cast(stride_po_s, tl.int64)
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64)

    po_base = PartialOutput + pid_t_64 * stride_po_t_64 + offs_h[:, None] * stride_po_h
    po_base_0 = po_base + 0 * stride_po_s_64
    po_base_1 = po_base + 1 * stride_po_s_64
    po_base_2 = po_base + 2 * stride_po_s_64
    po_base_3 = po_base + 3 * stride_po_s_64
    po_base_4 = po_base + 4 * stride_po_s_64
    po_base_5 = po_base + 5 * stride_po_s_64
    po_base_6 = po_base + 6 * stride_po_s_64
    po_base_7 = po_base + 7 * stride_po_s_64

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64 + offs_h[:, None] * stride_o_h

    # Loop over D dimension with BLOCK_D chunks
    num_d_iters: tl.constexpr = (512 + BLOCK_D - 1) // BLOCK_D
    for d_idx in tl.static_range(num_d_iters):
        d_offs = d_idx * BLOCK_D + offs_d[None, :]
        mask_d = d_offs < d_v
        mask_hd = mask_h[:, None] & mask_d

        po_0 = tl.load(po_base_0 + d_offs * stride_po_d, mask=mask_hd, other=0.0)
        po_1 = tl.load(po_base_1 + d_offs * stride_po_d, mask=mask_hd, other=0.0)
        po_2 = tl.load(po_base_2 + d_offs * stride_po_d, mask=mask_hd, other=0.0)
        po_3 = tl.load(po_base_3 + d_offs * stride_po_d, mask=mask_hd, other=0.0)
        po_4 = tl.load(po_base_4 + d_offs * stride_po_d, mask=mask_hd, other=0.0)
        po_5 = tl.load(po_base_5 + d_offs * stride_po_d, mask=mask_hd, other=0.0)
        po_6 = tl.load(po_base_6 + d_offs * stride_po_d, mask=mask_hd, other=0.0)
        po_7 = tl.load(po_base_7 + d_offs * stride_po_d, mask=mask_hd, other=0.0)

        combined = (
            scale_0[:, None] * po_0
            + scale_1[:, None] * po_1
            + scale_2[:, None] * po_2
            + scale_3[:, None] * po_3
            + scale_4[:, None] * po_4
            + scale_5[:, None] * po_5
            + scale_6[:, None] * po_6
            + scale_7[:, None] * po_7
        )
        tl.store(o_base + d_offs * stride_o_d, combined.to(tl.bfloat16), mask=mask_hd)

    stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)
    lse_ptrs = LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h
    tl.store(lse_ptrs, combined_lse, mask=mask_h)


@triton.jit
def _combine_splitk_kernel_2(
    PartialOutput,
    PartialLSE,
    AttnSink,
    Output,
    LSE,
    total_tokens,
    total_tokens_bucket,
    h_q,
    d_v,
    stride_po_s,
    stride_po_t,
    stride_po_h,
    stride_po_d,
    stride_plse_s,
    stride_plse_t,
    stride_plse_h,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_t,
    stride_lse_h,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Combine partial results from split-K kernel (SPLIT_K=2)."""
    LOG2E: tl.constexpr = 1.4426950408889634
    NEG_INF = float("-inf")
    POS_INF = float("+inf")
    INF_THRESHOLD = 1e30

    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    offs_d = tl.arange(0, BLOCK_D)

    stride_plse_s_64 = tl.cast(stride_plse_s, tl.int64)
    stride_plse_t_64 = tl.cast(stride_plse_t, tl.int64)

    lse_0 = tl.load(
        PartialLSE
        + 0 * stride_plse_s_64
        + pid_t_64 * stride_plse_t_64
        + offs_h * stride_plse_h,
        mask=mask_h,
        other=POS_INF,
    )
    lse_1 = tl.load(
        PartialLSE
        + 1 * stride_plse_s_64
        + pid_t_64 * stride_plse_t_64
        + offs_h * stride_plse_h,
        mask=mask_h,
        other=POS_INF,
    )

    lse_0_valid = tl.abs(lse_0) < INF_THRESHOLD
    lse_1_valid = tl.abs(lse_1) < INF_THRESHOLD

    lse_0_safe = tl.where(lse_0_valid, lse_0, NEG_INF)
    lse_1_safe = tl.where(lse_1_valid, lse_1, NEG_INF)

    max_lse = tl.maximum(lse_0_safe, lse_1_safe)

    exp_0 = tl.where(lse_0_valid, tl.math.exp2((lse_0_safe - max_lse) * LOG2E), 0.0)
    exp_1 = tl.where(lse_1_valid, tl.math.exp2((lse_1_safe - max_lse) * LOG2E), 0.0)

    sum_exp = exp_0 + exp_1
    all_invalid = sum_exp == 0.0
    sum_exp_safe = tl.where(all_invalid, 1.0, sum_exp)

    combined_lse = max_lse + tl.math.log2(sum_exp_safe) / LOG2E
    combined_lse = tl.where(all_invalid, POS_INF, combined_lse)

    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        is_lonely = combined_lse > INF_THRESHOLD
        lse_safe_for_sink = tl.where(is_lonely, 0.0, combined_lse)
        diff = attn_sink_vals - lse_safe_for_sink
        diff_clamped = tl.minimum(tl.maximum(diff, -100.0), 100.0)
        exp_diff = tl.math.exp2(diff_clamped * LOG2E)
        exp_diff = tl.where(is_lonely, 0.0, exp_diff)
        denominator = 1.0 + exp_diff
        sink_scale = 1.0 / denominator
        sink_scale = tl.where(is_lonely, 1.0, sink_scale)

        scale_0 = (exp_0 / sum_exp_safe) * sink_scale
        scale_1 = (exp_1 / sum_exp_safe) * sink_scale
    else:
        scale_0 = exp_0 / sum_exp_safe
        scale_1 = exp_1 / sum_exp_safe

    scale_0 = tl.where(all_invalid, 0.0, scale_0)
    scale_1 = tl.where(all_invalid, 0.0, scale_1)

    stride_po_s_64 = tl.cast(stride_po_s, tl.int64)
    stride_po_t_64 = tl.cast(stride_po_t, tl.int64)

    po_base_0 = (
        PartialOutput
        + 0 * stride_po_s_64
        + pid_t_64 * stride_po_t_64
        + offs_h[:, None] * stride_po_h
    )
    po_base_1 = (
        PartialOutput
        + 1 * stride_po_s_64
        + pid_t_64 * stride_po_t_64
        + offs_h[:, None] * stride_po_h
    )

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64 + offs_h[:, None] * stride_o_h

    for d_idx in range(4):
        d_offs = d_idx * BLOCK_D + offs_d[None, :]
        po_0 = tl.load(
            po_base_0 + d_offs * stride_po_d, mask=mask_h[:, None], other=0.0
        )
        po_1 = tl.load(
            po_base_1 + d_offs * stride_po_d, mask=mask_h[:, None], other=0.0
        )
        combined = scale_0[:, None] * po_0 + scale_1[:, None] * po_1
        tl.store(
            o_base + d_offs * stride_o_d, combined.to(tl.bfloat16), mask=mask_h[:, None]
        )

    stride_lse_t_64 = tl.cast(stride_lse_t, tl.int64)
    lse_ptrs = LSE + pid_t_64 * stride_lse_t_64 + offs_h * stride_lse_h
    tl.store(lse_ptrs, combined_lse, mask=mask_h)


def _select_split_k(topk: int, h_q: int, total_tokens: int = 64) -> int:
    """Select optimal split_k based on topk, h_q, and total_tokens.

    The split_k parameter controls how many parallel splits are used to process
    the topk dimension. Larger split_k increases parallelism but also increases
    the overhead of the combine kernel.

    Updated heuristics based on benchmarking with optimized BLOCK_N configs:
    - For large topk (>= 16384): split_k=4 provides good balance with existing combine kernel
    - For medium topk (8192-16383): split_k=4
    - For small topk (< 8192): split_k=2
    """
    if topk >= 8192:
        return 4
    else:
        return 2


# ============================================================================
# Low-overhead buffer pool for splitk operations
# ============================================================================
class SplitKBufferPool:
    """
    Pre-allocated buffer pool for split-K intermediate tensors.

    Caches partial_output and partial_lse buffers to avoid repeated allocations.
    Output buffers are always freshly allocated to ensure correctness.
    """

    _buffers = {}
    _device = None

    @classmethod
    def get_buffers(
        cls, split_k: int, total_tokens: int, h_q: int, d_v: int, device: torch.device
    ):
        """Get or create intermediate buffers for the given configuration."""
        key = (split_k, total_tokens, h_q, d_v, device)

        if key not in cls._buffers or cls._device != device:
            cls._device = device
            partial_output = torch.empty(
                split_k, total_tokens, h_q, d_v, dtype=torch.float32, device=device
            )
            partial_lse = torch.empty(
                split_k, total_tokens, h_q, dtype=torch.float32, device=device
            )

            cls._buffers[key] = {
                "partial_output": partial_output,
                "partial_lse": partial_lse,
                "stride_po": partial_output.stride(),
                "stride_plse": partial_lse.stride(),
            }

        return cls._buffers[key]

    @classmethod
    def clear(cls):
        """Clear all cached buffers."""
        cls._buffers.clear()
        cls._device = None


def fused_gather_attn_decode_dsv4_dual_scope_low_overhead(
    q: torch.Tensor,
    kv_cache_main: torch.Tensor,
    indices_main: torch.Tensor,
    block_size_main: int,
    kv_cache_extra: torch.Tensor,
    indices_extra: torch.Tensor,
    block_size_extra: int,
    sm_scale: float,
    topk_length_main: Optional[torch.Tensor] = None,
    topk_length_extra: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    s_q: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Low-overhead version of fused_gather_attn_decode_dsv4_dual_scope.

    This version uses pre-allocated intermediate buffers and cached strides
    to minimize Python overhead, which is significant for small batch sizes.

    The kernel computation is identical to the original version.
    Output buffers are always freshly allocated to ensure correctness.
    """
    total_tokens, h_q, d_qk = q.shape
    topk_main = indices_main.shape[1]
    topk_extra = indices_extra.shape[1]
    total_topk = topk_main + topk_extra
    d_v = DSV4_D_V
    device = q.device

    # Prepare main KV cache
    kv_uint8_main = kv_cache_main.view(torch.uint8)
    num_blocks_main = kv_cache_main.shape[0]
    stride_kv_block_main = kv_uint8_main.stride(0)
    kv_flat_main = kv_uint8_main.reshape(num_blocks_main, -1)

    # Prepare extra KV cache
    kv_uint8_extra = kv_cache_extra.view(torch.uint8)
    num_blocks_extra = kv_cache_extra.shape[0]
    stride_kv_block_extra = kv_uint8_extra.stride(0)
    kv_flat_extra = kv_uint8_extra.reshape(num_blocks_extra, -1)

    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        q = q.to(torch.bfloat16).contiguous()

    if not indices_main.is_contiguous():
        indices_main = indices_main.contiguous()
    if not indices_extra.is_contiguous():
        indices_extra = indices_extra.contiguous()

    # Determine split_k
    SPLITK_DUAL_SCOPE_TOPK_THRESHOLD = 2048
    use_splitk_for_small_bs = total_tokens <= 8 and (h_q >= 128 or total_topk >= 1024)
    use_splitk_for_h64_large_topk = (
        h_q <= 64 and total_topk >= 1024 and total_tokens > 8 and total_tokens <= 128
    )
    use_splitk_for_large_topk = (
        total_tokens > 64 and total_topk >= SPLITK_DUAL_SCOPE_TOPK_THRESHOLD
    )
    # For h_q > 64 (e.g. h_q=128), the non-splitk grid has very few blocks
    # in the H dimension (cdiv(128,64)=2), leading to low GPU utilization
    # at medium batch sizes.  Split-K doubles the parallelism.
    use_splitk_for_large_hq = (
        h_q > 64 and total_tokens > 8 and total_topk >= 256
    )

    if not (
        use_splitk_for_small_bs
        or use_splitk_for_h64_large_topk
        or use_splitk_for_large_topk
        or use_splitk_for_large_hq
    ):
        # Fall back to non-splitk version
        return fused_gather_attn_decode_dsv4_dual_scope(
            q,
            kv_cache_main,
            indices_main,
            block_size_main,
            kv_cache_extra,
            indices_extra,
            block_size_extra,
            sm_scale,
            topk_length_main,
            topk_length_extra,
            attn_sink,
            s_q,
        )

    # Select split_k based on workload and total_topk.
    # CUDA graph replay benchmarks show optimal split_k depends on both:
    #   - High topk (>=512, c4 layers): more splits needed to parallelize
    #   - Low topk (<512, c128 layers): fewer splits, less combine overhead
    if total_tokens <= 8:
        if total_topk >= 512 and total_tokens <= 4:
            # High topk + very small bs: split_k=8 is 8-33% faster than sk=4
            split_k = 8
        else:
            # split_k=4 gives 2x more blocks than split_k=2
            split_k = 4
    elif use_splitk_for_large_hq:
        # For h_q > 64 with bs > 8:
        if total_topk >= 512:
            # High topk: split_k=4 for all medium/large bs
            split_k = 4
        else:
            # Low topk: split_k=2 is sufficient
            split_k = 2
    elif use_splitk_for_h64_large_topk:
        split_k = 2
    else:
        split_k = _select_split_k(total_topk, h_q, total_tokens)

    topk_per_split = (total_topk + split_k - 1) // split_k

    # Get pre-allocated intermediate buffers
    buffers = SplitKBufferPool.get_buffers(split_k, total_tokens, h_q, d_v, device)
    partial_output = buffers["partial_output"]
    partial_lse = buffers["partial_lse"]
    stride_po = buffers["stride_po"]
    stride_plse = buffers["stride_plse"]

    # Reuse pre-allocated output buffers to avoid torch.empty() calls
    # that would be captured in CUDA graphs (each adds ~7-8us replay overhead).
    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

    # Prepare dummy tensors for optional parameters
    topk_length_main_tensor = (
        topk_length_main if topk_length_main is not None else lse[:1, 0]
    )
    topk_length_extra_tensor = (
        topk_length_extra if topk_length_extra is not None else lse[:1, 0]
    )
    attn_sink_tensor = attn_sink if attn_sink is not None else lse[0, :]

    # Pre-compute strides
    stride_q = q.stride()
    stride_o = output.stride()
    stride_lse = lse.stride()

    # Check if buffer ops should be disabled
    kv_cache_size_main = stride_kv_block_main * num_blocks_main
    kv_cache_size_extra = stride_kv_block_extra * num_blocks_extra
    disable_buffer_ops = (
        kv_cache_size_main > BUFFER_OPS_DISABLE_THRESHOLD
        or kv_cache_size_extra > BUFFER_OPS_DISABLE_THRESHOLD
    )

    # Grid for splitk kernel
    grid_splitk = lambda meta: (
        triton.cdiv(h_q, meta["BLOCK_H"]),
        total_tokens,
        split_k,
    )

    # Run splitk kernel
    if disable_buffer_ops:
        with triton.knobs.amd.scope():
            triton.knobs.amd.use_buffer_ops = False
            _fused_gather_attn_dsv4_dual_scope_splitk_kernel[grid_splitk](
                q,
                kv_flat_main,
                indices_main,
                topk_length_main_tensor,
                kv_flat_extra,
                indices_extra,
                topk_length_extra_tensor,
                partial_output,
                partial_lse,
                sm_scale,
                total_tokens,
                _bucket_total_tokens(total_tokens),
                h_q,
                topk_main,
                num_blocks_main,
                block_size_main,
                topk_extra,
                num_blocks_extra,
                block_size_extra,
                s_q,
                topk_per_split,
                stride_q[0],
                stride_q[1],
                stride_q[2],
                stride_kv_block_main,
                stride_kv_block_extra,
                indices_main.stride(0),
                indices_main.stride(1),
                indices_extra.stride(0),
                indices_extra.stride(1),
                stride_po[0],
                stride_po[1],
                stride_po[2],
                stride_po[3],
                stride_plse[0],
                stride_plse[1],
                stride_plse[2],
                HAS_TOPK_LENGTH_MAIN=topk_length_main is not None,
                HAS_TOPK_LENGTH_EXTRA=topk_length_extra is not None,
            )
    else:
        _fused_gather_attn_dsv4_dual_scope_splitk_kernel[grid_splitk](
            q,
            kv_flat_main,
            indices_main,
            topk_length_main_tensor,
            kv_flat_extra,
            indices_extra,
            topk_length_extra_tensor,
            partial_output,
            partial_lse,
            sm_scale,
            total_tokens,
            _bucket_total_tokens(total_tokens),
            h_q,
            topk_main,
            num_blocks_main,
            block_size_main,
            topk_extra,
            num_blocks_extra,
            block_size_extra,
            s_q,
            topk_per_split,
            stride_q[0],
            stride_q[1],
            stride_q[2],
            stride_kv_block_main,
            stride_kv_block_extra,
            indices_main.stride(0),
            indices_main.stride(1),
            indices_extra.stride(0),
            indices_extra.stride(1),
            stride_po[0],
            stride_po[1],
            stride_po[2],
            stride_po[3],
            stride_plse[0],
            stride_plse[1],
            stride_plse[2],
            HAS_TOPK_LENGTH_MAIN=topk_length_main is not None,
            HAS_TOPK_LENGTH_EXTRA=topk_length_extra is not None,
        )

    # Run combine kernel
    if split_k == 8:
        grid_combine = lambda meta: (total_tokens, triton.cdiv(h_q, meta["BLOCK_H"]))
        _combine_splitk_kernel_8_optimized[grid_combine](
            partial_output,
            partial_lse,
            attn_sink_tensor,
            output,
            lse,
            total_tokens,
            _bucket_total_tokens(total_tokens),
            h_q,
            d_v,
            stride_po[0],
            stride_po[1],
            stride_po[2],
            stride_po[3],
            stride_plse[0],
            stride_plse[1],
            stride_plse[2],
            stride_o[0],
            stride_o[1],
            stride_o[2],
            stride_lse[0],
            stride_lse[1],
            HAS_ATTN_SINK=attn_sink is not None,
        )
    else:
        BLOCK_H_COMBINE = 16
        BLOCK_D_COMBINE = 128
        grid_combine = (total_tokens, triton.cdiv(h_q, BLOCK_H_COMBINE))

        if split_k == 2:
            combine_kernel = _combine_splitk_kernel_2
        elif split_k == 4:
            combine_kernel = _combine_splitk_kernel
        else:
            raise ValueError(f"Unsupported split_k: {split_k}")

        combine_kernel[grid_combine](
            partial_output,
            partial_lse,
            attn_sink_tensor,
            output,
            lse,
            total_tokens,
            _bucket_total_tokens(total_tokens),
            h_q,
            d_v,
            stride_po[0],
            stride_po[1],
            stride_po[2],
            stride_po[3],
            stride_plse[0],
            stride_plse[1],
            stride_plse[2],
            stride_o[0],
            stride_o[1],
            stride_o[2],
            stride_lse[0],
            stride_lse[1],
            HAS_ATTN_SINK=attn_sink is not None,
            BLOCK_H=BLOCK_H_COMBINE,
            BLOCK_D=BLOCK_D_COMBINE,
            num_warps=4,
            num_stages=1,
        )

    return output, lse
