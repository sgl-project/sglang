from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

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
    if _is_hip and batch * seq_len == 0:
        return mixed_qkv, z, b, a
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
    V_POW2: tl.constexpr,
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
    # Base offsets of the v/z regions for head group i_qk. tl.arange only
    # accepts power-of-two extents, so non-power-of-two group sizes (e.g. the
    # v/k head ratio 3 of the dense 27B hybrids) walk the group one
    # HEAD_V-sized head at a time; power-of-two groups keep the single wide
    # vector access. V_POW2 arrives as a wrapper-computed constexpr so the
    # dead branch is pruned before tl.arange validation.
    v_ld_base = (
        mixed_qkvz + i_bs * TOTAL_QKVZ + TOTAL_Q + TOTAL_K + i_qk * V_PER_GROUP * HEAD_V
    )
    z_ld_base = v_ld_base + TOTAL_V

    # ── Write to output (identical layout to the interleaved kernel) ──
    blk_q_st_ptr = mixed_qkv + i_bs * QKV_DIM_T + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
    blk_k_st_ptr = (
        mixed_qkv
        + i_bs * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    v_st_base = (
        mixed_qkv
        + i_bs * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * V_PER_GROUP * HEAD_V
    )
    z_st_base = z + i_bs * NUM_HEADS_V * HEAD_V + i_qk * V_PER_GROUP * HEAD_V

    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    if V_POW2:
        offs_group = tl.arange(0, V_PER_GROUP * HEAD_V)
        tl.store(v_st_base + offs_group, tl.load(v_ld_base + offs_group))
        tl.store(z_st_base + offs_group, tl.load(z_ld_base + offs_group))
    else:
        offs_head = tl.arange(0, HEAD_V)
        for i in tl.static_range(V_PER_GROUP):
            tl.store(
                v_st_base + i * HEAD_V + offs_head,
                tl.load(v_ld_base + i * HEAD_V + offs_head),
            )
            tl.store(
                z_st_base + i * HEAD_V + offs_head,
                tl.load(z_ld_base + i * HEAD_V + offs_head),
            )

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
    if _is_hip and batch * seq_len == 0:
        return mixed_qkv, z, b, a
    v_per_group = num_heads_v // num_heads_qk
    grid = (batch * seq_len, num_heads_qk)
    # Each program moves `v_per_group * head_v` elements for both v and z. For
    # the small head-group ratios (<= 512 elements) a single warp is the best
    # fit; wider ratios (e.g. 8 v-heads per k-head) need more lanes so the
    # per-program vector load/store does not serialize. The threshold was tuned
    # on MI355X, so it is confined to the HIP/aiter path; every other backend
    # keeps the original `num_warps=1`.
    num_warps = 1
    if _use_aiter:
        v_elems_per_program = (num_heads_v // num_heads_qk) * head_v
        num_warps = 1 if v_elems_per_program <= 512 else 4
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
        V_POW2=(v_per_group & (v_per_group - 1)) == 0,
        num_warps=num_warps,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


# Fusion begins after the quantized GEMMs: qkvz=[q|k|v|z], ba=[b|a].
# This tail unpacks both projections and updates the causal Conv1D state.


@triton.jit
def _fused_qkvzba_causal_conv1d_update_contiguous_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    conv_state,
    conv_weight,
    conv_bias,
    conv_state_indices,
    stride_qkvz_batch: tl.constexpr,
    stride_qkvz_dim: tl.constexpr,
    stride_ba_batch: tl.constexpr,
    stride_ba_dim: tl.constexpr,
    stride_state_batch: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_pos: tl.constexpr,
    stride_weight_dim: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_state_indices: tl.constexpr,
    QKV_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_STATE_SLOTS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    qkv_mask = dim_idx < QKV_DIM

    x = tl.load(
        mixed_qkvz + batch_idx * stride_qkvz_batch + dim_idx * stride_qkvz_dim,
        mask=qkv_mask,
        other=0.0,
    )

    state_slot = tl.load(conv_state_indices + batch_idx * stride_state_indices).to(
        tl.int64
    )
    # Treat every out-of-range index as padding so stale replay metadata cannot
    # turn an indexed state update into an OOB access.
    valid_slot = (
        (state_slot != PAD_SLOT_ID) & (state_slot >= 0) & (state_slot < NUM_STATE_SLOTS)
    )
    state_base = (
        conv_state + state_slot * stride_state_batch + dim_idx * stride_state_dim
    )

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(conv_bias + dim_idx, mask=qkv_mask, other=0.0).to(tl.float32)

    # Match the deployed direct-Triton update exactly. Its effective decode
    # state length is width-1 even when the physical cache tensor is wider.
    for pos in tl.static_range(KERNEL_WIDTH - 1):
        state_value = tl.load(
            state_base + pos * stride_state_pos,
            mask=qkv_mask & valid_slot,
            other=0.0,
        )
        weight_value = tl.load(
            conv_weight + dim_idx * stride_weight_dim + pos * stride_weight_width,
            mask=qkv_mask,
            other=0.0,
        )
        # Do not force an FP32 multiply here. This expression deliberately
        # retains the operand types/order of causal_conv1d_triton.py.
        acc += state_value * weight_value

    last_weight = tl.load(
        conv_weight
        + dim_idx * stride_weight_dim
        + (KERNEL_WIDTH - 1) * stride_weight_width,
        mask=qkv_mask,
        other=0.0,
    )
    acc += x * last_weight
    if SILU_ACTIVATION:
        conv_out = acc / (1.0 + tl.exp(-acc))
    else:
        conv_out = acc

    # The legacy kernel leaves padded rows' input unchanged.
    conv_out = tl.where(valid_slot, conv_out, x)
    tl.store(
        mixed_qkv + batch_idx * QKV_DIM + dim_idx,
        conv_out,
        mask=qkv_mask,
    )

    # The direct-Triton wrapper sets effective state_len=width-1 for decode.
    for pos in tl.static_range(KERNEL_WIDTH - 2):
        next_value = tl.load(
            state_base + (pos + 1) * stride_state_pos,
            mask=qkv_mask & valid_slot,
            other=0.0,
        )
        tl.store(
            state_base + pos * stride_state_pos,
            next_value,
            mask=qkv_mask & valid_slot,
        )
    tl.store(
        state_base + (KERNEL_WIDTH - 2) * stride_state_pos,
        x,
        mask=qkv_mask & valid_slot,
    )

    # The first feature lanes also materialize the smaller downstream tensors.
    z_mask = dim_idx < V_DIM
    z_value = tl.load(
        mixed_qkvz
        + batch_idx * stride_qkvz_batch
        + (QKV_DIM + dim_idx) * stride_qkvz_dim,
        mask=z_mask,
        other=0.0,
    )
    tl.store(z + batch_idx * V_DIM + dim_idx, z_value, mask=z_mask)

    gate_mask = dim_idx < NUM_V_HEADS
    b_value = tl.load(
        mixed_ba + batch_idx * stride_ba_batch + dim_idx * stride_ba_dim,
        mask=gate_mask,
        other=0.0,
    )
    a_value = tl.load(
        mixed_ba
        + batch_idx * stride_ba_batch
        + (NUM_V_HEADS + dim_idx) * stride_ba_dim,
        mask=gate_mask,
        other=0.0,
    )
    tl.store(b + batch_idx * NUM_V_HEADS + dim_idx, b_value, mask=gate_mask)
    tl.store(a + batch_idx * NUM_V_HEADS + dim_idx, a_value, mask=gate_mask)


def can_use_fused_qkvzba_causal_conv1d_update_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state_indices: torch.Tensor,
    *,
    qkv_dim: int,
    v_dim: int,
    num_v_heads: int,
    activation: str | None,
) -> tuple[bool, str]:
    """Return an explicit eligibility decision for the decode fusion."""
    tensors = (mixed_qkvz, mixed_ba, conv_state, conv_weight, conv_state_indices)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False, "all inputs must be torch.Tensor instances"
    if not all(tensor.is_cuda for tensor in tensors):
        return False, "CUDA tensors are required"
    if mixed_qkvz.ndim != 2 or mixed_ba.ndim != 2:
        return False, "projection outputs must be rank-2"
    if conv_state.ndim != 3 or conv_weight.ndim != 2:
        return False, "Conv1D state/weight ranks must be 3/2"
    if conv_state_indices.ndim != 1:
        return False, "conv_state_indices must be rank-1"
    batch = mixed_qkvz.shape[0]
    if mixed_ba.shape[0] != batch or conv_state_indices.shape[0] != batch:
        return False, "batch dimensions must match"
    if qkv_dim <= 0 or v_dim <= 0 or num_v_heads <= 0:
        return False, "TP-local dimensions must be positive"
    if mixed_qkvz.shape[1] != qkv_dim + v_dim:
        return False, "qkvz layout is not contiguous [Q|K|V|Z]"
    if mixed_ba.shape[1] != 2 * num_v_heads:
        return False, "ba layout is not contiguous [B|A]"
    if conv_state.shape[1] != qkv_dim or conv_weight.shape[0] != qkv_dim:
        return False, "Conv1D feature dimension does not match packed QKV"
    width = conv_weight.shape[1]
    if width < 2 or width > 4:
        return False, "only Conv1D widths 2 through 4 are supported"
    if conv_state.shape[2] < width - 1:
        return False, "Conv1D state is shorter than width - 1"
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if mixed_qkvz.dtype not in supported_dtypes:
        return False, "QKVZ activation dtype must be FP16, BF16, or FP32"
    if conv_state.dtype != mixed_qkvz.dtype or conv_weight.dtype != mixed_qkvz.dtype:
        return False, "QKVZ, Conv1D state, and weight dtypes must match"
    if mixed_ba.dtype not in supported_dtypes:
        return False, "BA activation dtype must be FP16, BF16, or FP32"
    if conv_bias is not None:
        if (
            not isinstance(conv_bias, torch.Tensor)
            or not conv_bias.is_cuda
            or conv_bias.ndim != 1
            or conv_bias.shape[0] != qkv_dim
            or conv_bias.dtype != mixed_qkvz.dtype
        ):
            return False, "Conv1D bias contract is incompatible"
    if activation not in (None, "silu", "swish"):
        return False, "activation must be None, silu, or swish"
    if mixed_qkvz.stride(1) != 1 or mixed_ba.stride(1) != 1:
        return False, "projection feature dimensions must be contiguous"
    if conv_weight.stride(1) != 1:
        return False, "Conv1D weight width dimension must be contiguous"
    if conv_state_indices.dtype not in (torch.int32, torch.int64):
        return False, "conv_state_indices must be int32 or int64"
    return True, "eligible"


def fused_qkvzba_causal_conv1d_update_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state_indices: torch.Tensor,
    *,
    qkv_dim: int,
    v_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    activation: str | None,
    pad_slot_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode-only fused Qwen3.5 projection unpack and Conv1D state update."""
    eligible, reason = can_use_fused_qkvzba_causal_conv1d_update_contiguous(
        mixed_qkvz,
        mixed_ba,
        conv_state,
        conv_weight,
        conv_bias,
        conv_state_indices,
        qkv_dim=qkv_dim,
        v_dim=v_dim,
        num_v_heads=num_v_heads,
        activation=activation,
    )
    if not eligible:
        raise ValueError(f"Ineligible fused GDN decode projection/Conv1D: {reason}")
    if v_dim != num_v_heads * head_v_dim:
        raise ValueError(
            "Ineligible fused GDN decode projection/Conv1D: "
            "v_dim must equal num_v_heads * head_v_dim"
        )

    batch = mixed_qkvz.shape[0]
    mixed_qkv = torch.empty(
        (batch, qkv_dim), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
    )
    z = torch.empty(
        (batch, num_v_heads, head_v_dim),
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        (batch, num_v_heads),
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)

    block_size = 256
    grid = (batch, triton.cdiv(qkv_dim, block_size))
    _fused_qkvzba_causal_conv1d_update_contiguous_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        conv_state,
        conv_weight,
        conv_bias,
        conv_state_indices,
        mixed_qkvz.stride(0),
        mixed_qkvz.stride(1),
        mixed_ba.stride(0),
        mixed_ba.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(1),
        conv_state_indices.stride(0),
        QKV_DIM=qkv_dim,
        V_DIM=v_dim,
        NUM_V_HEADS=num_v_heads,
        NUM_STATE_SLOTS=conv_state.shape[0],
        STATE_LEN=conv_state.shape[2],
        KERNEL_WIDTH=conv_weight.shape[1],
        HAS_BIAS=conv_bias is not None,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        PAD_SLOT_ID=pad_slot_id,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=2,
    )
    return mixed_qkv, z, b, a


@triton.jit
def fused_qkv_split_gdn_prefill_kernel(
    q,
    k,
    v,
    mixed_qkv,
    MIXED_QKV_STRIDE_T: tl.constexpr,
    MIXED_QKV_STRIDE_D: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    HEAD_V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_t = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    q_dim: tl.constexpr = NUM_Q_HEADS * HEAD_Q
    k_dim: tl.constexpr = NUM_K_HEADS * HEAD_K
    v_dim: tl.constexpr = NUM_V_HEADS * HEAD_V
    qk_dim: tl.constexpr = q_dim + k_dim
    qkv_dim: tl.constexpr = qk_dim + v_dim

    mask = offsets < qkv_dim
    values = tl.load(
        mixed_qkv + i_t * MIXED_QKV_STRIDE_T + offsets * MIXED_QKV_STRIDE_D,
        mask=mask,
    )

    q_mask = offsets < q_dim
    tl.store(q + i_t * q_dim + offsets, values, mask=q_mask)

    k_offsets = offsets - q_dim
    k_mask = (offsets >= q_dim) & (offsets < qk_dim)
    tl.store(k + i_t * k_dim + k_offsets, values, mask=k_mask)

    v_offsets = offsets - qk_dim
    v_mask = (offsets >= qk_dim) & (offsets < qkv_dim)
    tl.store(v + i_t * v_dim + v_offsets, values, mask=v_mask)


def fused_qkv_split_gdn_prefill(
    mixed_qkv: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q: int,
    head_k: int,
    head_v: int,
):
    """Split packed post-conv GDN QKV into contiguous FLA prefill tensors.

    `mixed_qkv` is laid out per token as `[all_q | all_k | all_v]`. The FLA
    chunk kernels consume separate contiguous `[1, T, H, D]` tensors, so this
    fused split replaces three independent `aten::copy_` kernels from the
    generic FLA input guard. `mixed_qkv` may be a strided `[T, qkv_dim]` view.
    """
    seq_len = mixed_qkv.shape[0]
    q = torch.empty(
        (1, seq_len, num_q_heads, head_q),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    k = torch.empty(
        (1, seq_len, num_k_heads, head_k),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    v = torch.empty(
        (1, seq_len, num_v_heads, head_v),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )

    qkv_dim = num_q_heads * head_q + num_k_heads * head_k + num_v_heads * head_v
    if _is_hip and seq_len == 0:
        return q, k, v
    fused_qkv_split_gdn_prefill_kernel[(seq_len,)](
        q,
        k,
        v,
        mixed_qkv,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_q,
        head_k,
        head_v,
        BLOCK_SIZE=triton.next_power_of_2(qkv_dim),
        num_warps=8,
        num_stages=3,
    )
    return q, k, v
