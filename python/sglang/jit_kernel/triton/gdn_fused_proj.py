from __future__ import annotations

import torch
import triton
import triton.language as tl

# =============================================================================
# Fused kernel — reads INTERLEAVED input format
# Used by Qwen3-Next whose checkpoint stores fused in_proj_qkvz weights
# in per-head-group interleaved layout:
#   [g0_q, g0_k, g0_v, g0_z, g1_q, g1_k, g1_v, g1_z, ...]
# =============================================================================


@triton.jit
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    q_end: tl.constexpr = HEAD_QK
    blk_q_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(0, q_end)
    )
    k_end: tl.constexpr = q_end + HEAD_QK
    blk_k_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(q_end, k_end)
    )
    v_end: tl.constexpr = k_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_v_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(k_end, v_end)
    )
    z_end: tl.constexpr = v_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_z_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(v_end, z_end)
    )
    blk_q_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_k_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_v_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )
    blk_z_st_ptr = (
        z
        + i_bs * NUM_HEADS_V * HEAD_V
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )
    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))
    b_end: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V // NUM_HEADS_QK
    for i in tl.static_range(b_end):
        blk_b_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))
    for i in tl.static_range(b_end, a_end):
        blk_a_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_a_st_ptr = (
            a + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + (i - b_end)
        )
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    batch, seq_len = mixed_qkvz.shape[0], 1
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)
    grid = (batch * seq_len, num_heads_qk)
    fused_qkvzba_split_reshape_cat_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


# =============================================================================
# Fused kernel — reads CONTIGUOUS input format
# Used by Qwen3.5 whose checkpoint stores in_proj_qkv and in_proj_z separately.
# After MergedColumnParallelLinear loads them, the matmul output is contiguous:
#   mixed_qkvz: [all_q | all_k | all_v | all_z]
#   mixed_ba:   [all_b | all_a]
#
# Output format is identical to the interleaved kernel (same downstream consumer).
# =============================================================================


@triton.jit
def fused_qkvzba_split_reshape_cat_contiguous_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)

    V_PER_GROUP: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK

    # ── Input dimensions (contiguous layout) ──
    TOTAL_Q: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    TOTAL_K: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    TOTAL_V: tl.constexpr = NUM_HEADS_V * HEAD_V
    TOTAL_QKVZ: tl.constexpr = TOTAL_Q + TOTAL_K + TOTAL_V + TOTAL_V
    TOTAL_BA: tl.constexpr = NUM_HEADS_V * 2

    # ── Output dimensions ──
    QKV_DIM_T: tl.constexpr = TOTAL_Q + TOTAL_K + TOTAL_V

    # ── Read from contiguous input ──
    # q for head group i_qk: in the all_q region, offset i_qk * HEAD_QK
    blk_q_ptr = mixed_qkvz + i_bs * TOTAL_QKVZ + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
    # k for head group i_qk: in the all_k region
    blk_k_ptr = (
        mixed_qkvz
        + i_bs * TOTAL_QKVZ
        + TOTAL_Q
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    # v for head group i_qk: in the all_v region
    blk_v_ptr = (
        mixed_qkvz
        + i_bs * TOTAL_QKVZ
        + TOTAL_Q
        + TOTAL_K
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )
    # z for head group i_qk: in the all_z region
    blk_z_ptr = (
        mixed_qkvz
        + i_bs * TOTAL_QKVZ
        + TOTAL_Q
        + TOTAL_K
        + TOTAL_V
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )

    # ── Write to output (identical layout to the interleaved kernel) ──
    blk_q_st_ptr = mixed_qkv + i_bs * QKV_DIM_T + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
    blk_k_st_ptr = (
        mixed_qkv
        + i_bs * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_v_st_ptr = (
        mixed_qkv
        + i_bs * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )
    blk_z_st_ptr = (
        z
        + i_bs * NUM_HEADS_V * HEAD_V
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )

    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))

    # ── b and a from contiguous [all_b | all_a] ──
    for i in tl.static_range(V_PER_GROUP):
        blk_b_ptr = mixed_ba + i_bs * TOTAL_BA + i_qk * V_PER_GROUP + i
        blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))

    for i in tl.static_range(V_PER_GROUP):
        blk_a_ptr = mixed_ba + i_bs * TOTAL_BA + NUM_HEADS_V + i_qk * V_PER_GROUP + i
        blk_a_st_ptr = a + i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    """Fused split/reshape/cat for CONTIGUOUS input format (Qwen3.5).

    Input layout:
        mixed_qkvz: [all_q | all_k | all_v | all_z]
        mixed_ba:   [all_b | all_a]

    Output layout (same as fused_qkvzba_split_reshape_cat):
        mixed_qkv: [all_q | all_k | all_v]  (z stripped)
        z: [num_v_heads, head_v]
        b: [num_v_heads]
        a: [num_v_heads]
    """
    batch, seq_len = mixed_qkvz.shape[0], 1
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)
    grid = (batch * seq_len, num_heads_qk)
    fused_qkvzba_split_reshape_cat_contiguous_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a
