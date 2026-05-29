"""Rank-specialized LoRA-B expand for virtual-expert LoRA.

The kernel here was originally a chunk in ``lora/triton_ops/virtual_experts.py``.
It is rank-specialized: the ``R`` dimension (LoRA rank) is a triton
``constexpr``, so each rank value used at runtime gets its own JIT-compiled
specialization (R=16, R=32, R=64 are all supported up to the ``R <= 64`` assert,
with no perf interaction between them — each gets its own kernel).

Called from :mod:`sglang.srt.lora.triton_ops.virtual_experts` when
``use_direct_expand_add=True`` (the trtllm-lora path uses this when
``max_lora_rank <= 64``); the generic ``invoke_fused_moe_kernel`` is used
when that flag is False (incl. ranks above 64).
"""
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_lora_expand_add_kernel(
    # Pointers
    a_ptr,  # [num_tokens * top_k, rank]
    b_ptr,  # [num_virtual_experts, N, rank]
    c_ptr,  # [num_tokens, N]
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions
    N,
    R: tl.constexpr,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ar,
    stride_be,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    # Constexprs
    router_topk: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    FUSE_SUM_ALL_REDUCE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Rank-specialized LoRA-B expand for virtual-expert LoRA."""
    pid = tl.program_id(0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        if not FUSE_SUM_ALL_REDUCE:
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            c_ptrs = (
                c_ptr
                + offs_token[:, None] * stride_cm
                + offs_n[None, :] * stride_cn
            )
            c_mask = token_mask[:, None] & (offs_n[None, :] < N)
            zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.dtype.element_ty)
            tl.store(c_ptrs, zeros, mask=c_mask)
        return

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_r = tl.arange(0, BLOCK_SIZE_R).to(tl.int64)
    rank_mask = offs_r < R

    a = tl.load(
        a_ptr + offs_token[:, None] * stride_am + offs_r[None, :] * stride_ar,
        mask=token_mask[:, None] & rank_mask[None, :],
        other=0.0,
    )
    b = tl.load(
        b_ptr
        + off_expert * stride_be
        + offs_n[None, :] * stride_bn
        + offs_r[:, None] * stride_br,
        mask=(offs_n[None, :] < N) & rank_mask[:, None],
        other=0.0,
    )

    accumulator = tl.dot(a, b, out_dtype=tl.float32)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    if FUSE_SUM_ALL_REDUCE:
        offs_token_out = offs_token // router_topk
    else:
        offs_token_out = offs_token
    c_ptrs = (
        c_ptr
        + offs_token_out[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)
    if FUSE_SUM_ALL_REDUCE:
        tl.atomic_add(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


def _invoke_moe_lora_expand_add(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: "dict[str, Any]",
    mul_routed_weight: bool,
    fuse_sum_all_reduce: bool,
) -> None:
    """Launch the rank-specialized LoRA-B expand kernel.

    ``R`` (= ``weight.shape[2]``) up to 64 is supported. ``BLOCK_SIZE_R`` is
    set to ``next_power_of_2(R)`` so each rank value pairs with the smallest
    tile that covers it (R=16 → BSR=16, R=32 → BSR=32, R=64 → BSR=64).
    Triton compiles a separate specialization per (R, BLOCK_SIZE_R) combo
    so different ranks don't interfere with each other's perf.
    """
    N = weight.shape[1]
    R = weight.shape[2]
    assert R <= 64, f"direct LoRA expand/add expects rank <= 64, got {R}"

    block_size_m = config["BLOCK_SIZE_M"]
    block_size_n = 128 if N % 128 == 0 else config["BLOCK_SIZE_N"]
    group_size_m = config.get("GROUP_SIZE_M", 1)
    block_size_r = triton.next_power_of_2(R)

    grid = (
        triton.cdiv(sorted_token_ids.shape[0], block_size_m)
        * triton.cdiv(N, block_size_n),
    )

    _moe_lora_expand_add_kernel[grid](
        intermediate,
        weight,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        R,
        topk_ids.numel(),
        intermediate.stride(0),
        intermediate.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(-2),
        output.stride(-1),
        router_topk=topk_ids.shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        FUSE_SUM_ALL_REDUCE=fuse_sum_all_reduce,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_R=block_size_r,
        GROUP_SIZE_M=group_size_m,
        num_warps=config.get("num_warps", 4),
        num_stages=1,
    )
