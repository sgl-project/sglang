"""Shared-outer reduction primitives for Marlin MoE LoRA.

For Inkling adapters, gate/up LoRA-A and down LoRA-B are shared by every
routed expert.  The gate shrink can therefore run once per token, while the
down rank vectors can be gamma-weighted and summed before the shared expand.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _weighted_topk_rank_sum_kernel(
    routed_rank_ptr,  # [M, topk, R]
    topk_weights_ptr,  # [M, topk]
    output_ptr,  # [M, R]
    M,
    R,
    scale,
    stride_rm,
    stride_rk,
    stride_rr,
    stride_wm,
    stride_wk,
    stride_om,
    stride_or,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rank_block = tl.program_id(1)
    rank = rank_block * BLOCK_R + tl.arange(0, BLOCK_R)
    token_mask = token < M
    rank_mask = rank < R
    mask = token_mask[:, None] & rank_mask[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        routed_rank = tl.load(
            routed_rank_ptr
            + token[:, None] * stride_rm
            + k * stride_rk
            + rank[None, :] * stride_rr,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            topk_weights_ptr + token * stride_wm + k * stride_wk,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += routed_rank * weight[:, None]

    accumulator *= scale
    tl.store(
        output_ptr + token[:, None] * stride_om + rank[None, :] * stride_or,
        accumulator,
        mask=mask,
    )


def weighted_topk_rank_sum(
    routed_rank: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float,
    *,
    block_m: int,
) -> None:
    """Compute ``output[m] = scale * sum_k(gamma[m,k] * rank[m,k])``."""

    assert routed_rank.ndim == 3
    assert topk_weights.shape == routed_rank.shape[:2]
    assert output.shape == (routed_rank.shape[0], routed_rank.shape[2])
    assert routed_rank.is_contiguous() and topk_weights.is_contiguous()
    assert output.is_contiguous()
    if routed_rank.shape[0] == 0:
        return

    rank = routed_rank.shape[2]
    assert block_m in (1, 2)
    block_rank = max(16, triton.next_power_of_2(rank))
    _weighted_topk_rank_sum_kernel[
        (triton.cdiv(routed_rank.shape[0], block_m), triton.cdiv(rank, block_rank))
    ](
        routed_rank,
        topk_weights,
        output,
        routed_rank.shape[0],
        rank,
        routed_scaling_factor,
        routed_rank.stride(0),
        routed_rank.stride(1),
        routed_rank.stride(2),
        topk_weights.stride(0),
        topk_weights.stride(1),
        output.stride(0),
        output.stride(1),
        TOPK=routed_rank.shape[1],
        BLOCK_M=block_m,
        BLOCK_R=block_rank,
        num_warps=1,
    )


@triton.jit
def _fused_base_shared_lora_reduce_kernel(
    routed_base_ptr,  # [M, topk, K], already weighted by the base MoE
    routed_rank_ptr,  # [M, topk, R]
    topk_weights_ptr,  # [M, topk]
    shared_b_ptr,  # [K, R]
    output_ptr,  # [M, K]
    M,
    K,
    R,
    scale,
    stride_bm,
    stride_bk,
    stride_bn,
    stride_rm,
    stride_rk,
    stride_rr,
    stride_wm,
    stride_wk,
    stride_sk,
    stride_sr,
    stride_om,
    stride_ok,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Fuse base top-k reduction with the shared-B LoRA decode tail."""

    token = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    out_col = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    rank = tl.arange(0, BLOCK_R)
    token_mask = token < M
    out_mask = out_col < K
    rank_mask = rank < R

    base_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    rank_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    for topk_idx in tl.static_range(TOPK):
        base = tl.load(
            routed_base_ptr
            + token[:, None] * stride_bm
            + topk_idx * stride_bk
            + out_col[None, :] * stride_bn,
            mask=token_mask[:, None] & out_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        routed_rank = tl.load(
            routed_rank_ptr
            + token[:, None] * stride_rm
            + topk_idx * stride_rk
            + rank[None, :] * stride_rr,
            mask=token_mask[:, None] & rank_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            topk_weights_ptr + token * stride_wm + topk_idx * stride_wk,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        base_acc += base
        rank_acc += routed_rank * weight[:, None]

    # Match the existing path's BF16 rank-sum materialization before the
    # tensor-core shared-B GEMM.  This also keeps the dot operands type-aligned.
    scaled_rank = (rank_acc * scale).to(shared_b_ptr.dtype.element_ty)
    shared_b = tl.load(
        shared_b_ptr + out_col[None, :] * stride_sk + rank[:, None] * stride_sr,
        mask=out_mask[None, :] & rank_mask[:, None],
        other=0.0,
    )
    lora_acc = tl.dot(scaled_rank, shared_b, out_dtype=tl.float32)
    # The existing reducer materializes the scaled base sum in the BF16 output
    # before cuBLAS addmm reads it back. Preserve that rounding point so fusion
    # changes launch structure, not the numerical contract.
    scaled_base = (base_acc * scale).to(output_ptr.dtype.element_ty).to(tl.float32)
    result = scaled_base + lora_acc
    tl.store(
        output_ptr + token[:, None] * stride_om + out_col[None, :] * stride_ok,
        result,
        mask=token_mask[:, None] & out_mask[None, :],
    )


def fused_base_shared_lora_reduce(
    routed_base: torch.Tensor,
    routed_rank: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_b: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float,
    *,
    block_m: int,
    block_k: int,
) -> None:
    """Reduce base experts and add a factored shared-B LoRA delta in one pass.

    This is intended for decode-sized batches, where replacing three
    latency-bound launches is more important than maximizing the standalone
    shared-B GEMM throughput.  ``routed_base`` must already contain the router
    weights, matching Marlin's ``mul_topk_weights=True`` output.
    """

    assert routed_base.ndim == 3 and routed_rank.ndim == 3
    assert routed_base.shape[:2] == routed_rank.shape[:2] == topk_weights.shape
    assert shared_b.shape == (routed_base.shape[2], routed_rank.shape[2])
    assert output.shape == (routed_base.shape[0], routed_base.shape[2])
    assert 0 < routed_rank.shape[2] <= 64
    assert routed_base.is_cuda
    assert (
        routed_base.device
        == routed_rank.device
        == topk_weights.device
        == shared_b.device
        == output.device
    )
    assert routed_base.dtype in (torch.bfloat16, torch.float16)
    assert routed_base.dtype == routed_rank.dtype == shared_b.dtype == output.dtype
    assert topk_weights.dtype == torch.float32
    assert routed_base.is_contiguous() and routed_rank.is_contiguous()
    assert topk_weights.is_contiguous() and shared_b.is_contiguous()
    assert output.is_contiguous()
    assert all(
        output.data_ptr() != operand.data_ptr()
        for operand in (routed_base, routed_rank, topk_weights, shared_b)
    ), "fused shared-outer output must not alias an input"
    if routed_base.shape[0] == 0:
        return

    rank = routed_rank.shape[2]
    assert block_m in (1, 2, 4, 8)
    assert block_k in (32, 64, 128)
    block_rank = max(16, triton.next_power_of_2(rank))
    _fused_base_shared_lora_reduce_kernel[
        (
            triton.cdiv(routed_base.shape[0], block_m),
            triton.cdiv(routed_base.shape[2], block_k),
        )
    ](
        routed_base,
        routed_rank,
        topk_weights,
        shared_b,
        output,
        routed_base.shape[0],
        routed_base.shape[2],
        rank,
        routed_scaling_factor,
        routed_base.stride(0),
        routed_base.stride(1),
        routed_base.stride(2),
        routed_rank.stride(0),
        routed_rank.stride(1),
        routed_rank.stride(2),
        topk_weights.stride(0),
        topk_weights.stride(1),
        shared_b.stride(0),
        shared_b.stride(1),
        output.stride(0),
        output.stride(1),
        TOPK=routed_base.shape[1],
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_R=block_rank,
        num_warps=4 if block_k == 128 else 2,
        num_stages=1,
    )


# Keep a token-mapped kernel separate from the block-M single-slot kernel: the
# Decode uses a separate launch geometry.
@triton.jit
def _fused_base_mapped_shared_lora_reduce_kernel(
    routed_base_ptr,
    routed_rank_ptr,
    topk_weights_ptr,
    shared_b_ptr,  # [S, K, R]
    token_lora_mapping_ptr,
    output_ptr,
    K,
    R,
    scale,
    stride_bm,
    stride_bk,
    stride_bn,
    stride_rm,
    stride_rk,
    stride_rr,
    stride_wm,
    stride_wk,
    stride_ss,
    stride_sk,
    stride_sr,
    stride_om,
    stride_ok,
    TOPK: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    token = tl.program_id(0)
    out_col = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    rank = tl.arange(0, BLOCK_R)
    out_mask = out_col < K
    rank_mask = rank < R

    base_acc = tl.zeros((1, BLOCK_K), dtype=tl.float32)
    rank_acc = tl.zeros((1, BLOCK_R), dtype=tl.float32)
    for topk_idx in tl.static_range(TOPK):
        base = tl.load(
            routed_base_ptr
            + token * stride_bm
            + topk_idx * stride_bk
            + out_col[None, :] * stride_bn,
            mask=out_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        routed_rank = tl.load(
            routed_rank_ptr
            + token * stride_rm
            + topk_idx * stride_rk
            + rank[None, :] * stride_rr,
            mask=rank_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(topk_weights_ptr + token * stride_wm + topk_idx * stride_wk)
        base_acc += base
        rank_acc += routed_rank * weight

    slot = tl.load(token_lora_mapping_ptr + token).to(tl.int64)
    active = (slot >= 0) & (slot < NUM_SLOTS)
    scaled_rank = (rank_acc * scale).to(shared_b_ptr.dtype.element_ty)
    shared_b = tl.load(
        shared_b_ptr
        + slot * stride_ss
        + out_col[None, :] * stride_sk
        + rank[:, None] * stride_sr,
        mask=active & out_mask[None, :] & rank_mask[:, None],
        other=0.0,
    )
    lora_acc = tl.dot(scaled_rank, shared_b, out_dtype=tl.float32)
    scaled_base = (base_acc * scale).to(output_ptr.dtype.element_ty).to(tl.float32)
    tl.store(
        output_ptr + token * stride_om + out_col[None, :] * stride_ok,
        scaled_base + lora_acc,
        mask=out_mask[None, :],
    )


def fused_base_mapped_shared_lora_reduce(
    routed_base: torch.Tensor,
    routed_rank: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_b: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float,
    *,
    block_k: int,
) -> None:
    """Fused decode tail with a per-token shared-B adapter slot."""

    assert routed_base.ndim == routed_rank.ndim == 3
    assert routed_base.shape[:2] == routed_rank.shape[:2] == topk_weights.shape
    assert shared_b.ndim == 4 and shared_b.shape[1] == 1
    assert shared_b.shape[0] >= 2
    shared_b_view = shared_b[:, 0]
    assert shared_b_view.shape[1:] == (routed_base.shape[2], routed_rank.shape[2])
    assert token_lora_mapping.shape == (routed_base.shape[0],)
    assert token_lora_mapping.dtype == torch.int32
    assert output.shape == (routed_base.shape[0], routed_base.shape[2])
    assert 0 < routed_rank.shape[2] <= 64
    assert routed_base.dtype in (torch.bfloat16, torch.float16)
    assert routed_base.dtype == routed_rank.dtype == shared_b.dtype == output.dtype
    assert topk_weights.dtype == torch.float32
    assert all(
        tensor.device == routed_base.device
        for tensor in (
            routed_rank,
            topk_weights,
            shared_b,
            token_lora_mapping,
            output,
        )
    )
    assert all(
        tensor.is_cuda and tensor.is_contiguous()
        for tensor in (
            routed_base,
            routed_rank,
            topk_weights,
            shared_b,
            token_lora_mapping,
            output,
        )
    )
    assert all(
        output.data_ptr() != operand.data_ptr()
        for operand in (
            routed_base,
            routed_rank,
            topk_weights,
            shared_b,
            token_lora_mapping,
        )
    ), "mapped shared-outer output must not alias an input"
    if routed_base.shape[0] == 0:
        return

    assert block_k in (32, 64, 128)
    block_rank = max(16, triton.next_power_of_2(routed_rank.shape[2]))
    _fused_base_mapped_shared_lora_reduce_kernel[
        (routed_base.shape[0], triton.cdiv(routed_base.shape[2], block_k))
    ](
        routed_base,
        routed_rank,
        topk_weights,
        shared_b_view,
        token_lora_mapping,
        output,
        routed_base.shape[2],
        routed_rank.shape[2],
        routed_scaling_factor,
        routed_base.stride(0),
        routed_base.stride(1),
        routed_base.stride(2),
        routed_rank.stride(0),
        routed_rank.stride(1),
        routed_rank.stride(2),
        topk_weights.stride(0),
        topk_weights.stride(1),
        shared_b_view.stride(0),
        shared_b_view.stride(1),
        shared_b_view.stride(2),
        output.stride(0),
        output.stride(1),
        TOPK=routed_base.shape[1],
        NUM_SLOTS=shared_b.shape[0],
        BLOCK_K=block_k,
        BLOCK_R=block_rank,
        num_warps=4 if block_k == 128 else 2,
        num_stages=1,
    )


def fused_base_shared_lora_reduce_config(num_tokens: int) -> tuple[int, int]:
    """Return the tuned B200 launch geometry for M in ``[1, 512]``."""

    if not 0 < num_tokens <= 512:
        raise ValueError(
            f"fused shared-outer tail expects M in [1, 512], got {num_tokens}"
        )
    if num_tokens <= 2:
        return 1, 64 if num_tokens == 1 else 32
    if num_tokens <= 4:
        return 4, 32
    return 8, 64 if num_tokens <= 64 else 128
