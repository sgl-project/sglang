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

from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs


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
    GATED_A_HALF: tl.constexpr,
):
    """Rank-specialized LoRA-B expand for virtual-expert LoRA.

    ``GATED_A_HALF`` > 0 enables the gate/up split for a gated (SwiGLU) gate_up
    LoRA: the intermediate (A) has ``2*R`` columns (gate-shrink ``[0:R]`` then
    up-shrink ``[R:2R]``) and the output has ``2*GATED_A_HALF`` columns (gate
    then up). Output tiles in the up half (column >= ``GATED_A_HALF``) read the
    up-shrink columns ``[R:2R]`` instead of ``[0:R]``. ``GATED_A_HALF`` must be
    a multiple of ``BLOCK_SIZE_N`` so no tile straddles the gate/up boundary.
    ``GATED_A_HALF == 0`` is the non-gated path (read ``[0:R]`` for all tiles).
    """
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
                c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
            )
            c_mask = token_mask[:, None] & (offs_n[None, :] < N)
            zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.dtype.element_ty)
            tl.store(c_ptrs, zeros, mask=c_mask)
        return

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_r = tl.arange(0, BLOCK_SIZE_R).to(tl.int64)
    rank_mask = offs_r < R

    # Gated gate_up split: the up-half output tiles read up-shrink (A columns [R:2R]); gate-half
    # tiles read gate-shrink (A columns [0:R]). GATED_A_HALF == 0 -> always read [0:R] (non-gated).
    a_col = offs_r
    if GATED_A_HALF > 0:
        a_col = offs_r + tl.where(pid_n * BLOCK_SIZE_N >= GATED_A_HALF, R, 0)

    a = tl.load(
        a_ptr + offs_token[:, None] * stride_am + a_col[None, :] * stride_ar,
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
    c_ptrs = c_ptr + offs_token_out[:, None] * stride_cm + offs_n[None, :] * stride_cn
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
    force_block_size_n: "int | None" = None,
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
    # BLOCK_SIZE_N defaults to 128 when N % 128 == 0 (a good N-divisible tile that also keeps
    # the gated gate_up split aligned). ``force_block_size_n`` lets a tuner/bench override it;
    # down-proj has no gate/up boundary, so any divisor of N is valid there.
    if force_block_size_n is not None:
        block_size_n = force_block_size_n
    else:
        block_size_n = 128 if N % 128 == 0 else config["BLOCK_SIZE_N"]
    group_size_m = config.get("GROUP_SIZE_M", 1)
    block_size_r = triton.next_power_of_2(R)

    # gate_up LoRA: the shrink stacks gate_A and up_A, so the intermediate has 2*R columns
    # ([0:R] = gate-shrink x@gate_A^T, [R:2R] = up-shrink x@up_A^T). The up output half
    # (column >= N/2) must contract the up-shrink [R:2R], not gate_A's [0:R]. Reading [0:R]
    # for both halves (the previous hardcode) computed the up delta from gate_A and dropped
    # up_A -- wrong whenever gate_A != up_A (the normal independently-trained gate/up case;
    # verified >100% rel error vs a PEFT reference on the real Qwen3.5 adapter). The earlier
    # "vs cutlass" justification for reading [0:R] was unreliable (the cutlass reference shared
    # the same bug). Detect the gated layout from the intermediate width and split in-kernel.
    inter_width = intermediate.shape[1]
    assert inter_width in (R, 2 * R), (
        f"LoRA expand intermediate width must be R ({R}, non-gated) or 2*R "
        f"({2 * R}, gated gate_up), got {inter_width}"
    )
    # Lazy import to avoid the trtllm_moe <-> triton_ops package import cycle at load time.

    gated = inter_width == 2 * R
    use_gated_split = (
        gated and lora_envs.SGLANG_ENABLE_LORA_MOE_GATEUP_GATED_SPLIT.get()
    )
    gated_a_half = (N // 2) if use_gated_split else 0
    if use_gated_split:
        assert N % 2 == 0 and (N // 2) % block_size_n == 0, (
            f"gated gate_up split needs N/2 ({N // 2}) divisible by BLOCK_SIZE_N "
            f"({block_size_n})"
        )

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
        GATED_A_HALF=gated_a_half,
        num_warps=config.get("num_warps", 4),
        num_stages=1,
    )
