from typing import (
    Optional,
)

import torch
import triton
import triton.language as tl


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=32, num_stages=1),
        triton.Config({}, num_warps=32, num_stages=2),
    ],
    key=["num_tokens", "num_experts", "has_correction_bias"],
)
@triton.jit
def topk_softmax_triton_kernel(
    gating_output_ptr,
    selected_expert_ptr,
    moe_weights_ptr,
    renormalize_flag,
    num_experts,
    num_tokens,  # for autotune key
    moe_softcapping,
    correction_bias_ptr,
    has_correction_bias: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_WIDTH_SIZE_UP: tl.constexpr,
):
    curr_row_idx = tl.program_id(0)

    FLOAT_MINIMUM = -10000.0
    LOG2E = 1.4426950408889634

    weights_local_final = tl.zeros((BLOCK_K,), dtype=tl.float32)
    selected_local_final = tl.zeros((BLOCK_K,), dtype=tl.int32)

    offset = tl.arange(0, BLOCK_WIDTH_SIZE_UP)
    k_offset = tl.arange(0, BLOCK_K)
    mask_expert = offset < num_experts
    mask_topk = k_offset < K

    row_offset = curr_row_idx * num_experts

    logits = tl.load(
        gating_output_ptr + row_offset + offset, mask=mask_expert, other=FLOAT_MINIMUM
    )
    logits = tl.cast(logits, tl.float32)

    if has_correction_bias:
        bias = tl.load(correction_bias_ptr + offset, mask=mask_expert, other=0.0)
        logits = logits + bias

    if moe_softcapping > 0.0:
        logits = moe_softcapping * tanh(logits / moe_softcapping)

    row_max = tl.max(logits, axis=0)
    probs = tl.exp2((logits - row_max) * LOG2E)
    row_sum = tl.sum(probs, axis=0)
    inv_row_sum = 1.0 / row_sum
    probs = probs * inv_row_sum
    probs = tl.where(mask_expert, probs, FLOAT_MINIMUM)

    weights_selected_sum = 0.0
    for k_idx in range(K):
        top_k_index = tl.argmax(probs, axis=0)
        mask = offset == top_k_index
        top_k_value = tl.sum(tl.where(mask, probs, 0.0))

        weights_local_final = tl.where(
            k_offset == k_idx, top_k_value, weights_local_final
        )
        selected_local_final = tl.where(
            k_offset == k_idx, top_k_index, selected_local_final
        )
        weights_selected_sum += top_k_value

        probs = tl.where(offset == top_k_index, FLOAT_MINIMUM, probs)

    if renormalize_flag:
        weights_local_final = weights_local_final / weights_selected_sum

    tl.store(
        moe_weights_ptr + curr_row_idx * K + k_offset,
        weights_local_final,
        mask=mask_topk,
    )
    tl.store(
        selected_expert_ptr + curr_row_idx * K + k_offset,
        selected_local_final,
        mask=mask_topk,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    """
    Compute top-k softmax for MoE routing.

    Args:
        topk_weights: Output tensor for top-k weights [num_tokens, topk]
        topk_ids: Output tensor for top-k expert indices [num_tokens, topk]
        gating_output: Gating logits [num_tokens, num_experts]
        renormalize: Whether to renormalize the top-k weights
        moe_softcapping: Tanh softcapping value (0.0 to disable)
        correction_bias: Per-expert bias correction [num_experts], must be float32 if provided
    """

    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[-1]
    has_correction_bias = correction_bias is not None

    block_width_up = triton.next_power_of_2(num_experts)
    grid = (num_tokens,)

    topk_softmax_triton_kernel[grid](
        gating_output,
        topk_ids,
        topk_weights,
        renormalize,
        num_experts,
        num_tokens,
        moe_softcapping,
        correction_bias,
        has_correction_bias,
        K=topk,
        BLOCK_K=triton.next_power_of_2(topk),
        BLOCK_WIDTH_SIZE_UP=block_width_up,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=32, num_stages=1),
        triton.Config({}, num_warps=32, num_stages=2),
    ],
    key=["num_tokens", "num_experts"],
)
@triton.jit
def topk_sigmoid_triton_kernel(
    gating_output_ptr,
    selected_expert_ptr,
    moe_weights_ptr,
    renormalize_flag,
    correction_bias_ptr,
    has_correction_bias: tl.constexpr,
    num_experts,
    num_tokens,  # for autotune key
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_WIDTH_SIZE_UP: tl.constexpr,
):
    curr_row_idx = tl.program_id(0)

    FLOAT_MINIMUM = -10000.0
    LOG2E = 1.4426950408889634

    weights_local_final = tl.zeros((BLOCK_K,), dtype=tl.float32)
    selected_local_final = tl.zeros((BLOCK_K,), dtype=tl.int32)

    offset = tl.arange(0, BLOCK_WIDTH_SIZE_UP)
    k_offset = tl.arange(0, BLOCK_K)
    mask_expert = offset < num_experts
    mask_topk = k_offset < K

    row_offset = curr_row_idx * num_experts

    x = tl.load(
        gating_output_ptr + row_offset + offset, mask=mask_expert, other=FLOAT_MINIMUM
    )
    x = tl.cast(x, tl.float32)

    # Compute sigmoid(x)
    is_positive = x >= 0
    neg_x = tl.where(is_positive, -x, x)
    exp_neg_x = tl.exp2(neg_x * LOG2E)
    probs = tl.where(
        is_positive,
        1.0 / (1.0 + exp_neg_x),
        exp_neg_x / (1.0 + exp_neg_x),
    )

    if has_correction_bias:
        bias = tl.load(correction_bias_ptr + offset, mask=mask_expert, other=0.0)
        probs_for_choice = probs + bias
    else:
        probs_for_choice = probs

    probs_for_choice = tl.where(mask_expert, probs_for_choice, FLOAT_MINIMUM)

    weights_selected_sum = 0.0
    for k_idx in range(K):
        top_k_index = tl.argmax(probs_for_choice, axis=0)
        mask = offset == top_k_index
        top_k_value = tl.sum(tl.where(mask, probs, 0.0))

        weights_local_final = tl.where(
            k_offset == k_idx, top_k_value, weights_local_final
        )
        selected_local_final = tl.where(
            k_offset == k_idx, top_k_index, selected_local_final
        )
        weights_selected_sum += top_k_value

        probs_for_choice = tl.where(
            offset == top_k_index, FLOAT_MINIMUM, probs_for_choice
        )

    if renormalize_flag:
        weights_local_final = weights_local_final / weights_selected_sum

    tl.store(
        moe_weights_ptr + curr_row_idx * K + k_offset,
        weights_local_final,
        mask=mask_topk,
    )
    tl.store(
        selected_expert_ptr + curr_row_idx * K + k_offset,
        selected_local_final,
        mask=mask_topk,
    )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    """
    Compute top-k sigmoid for MoE routing.

    Args:
        topk_weights: Output tensor for top-k weights [num_tokens, topk]
        topk_ids: Output tensor for top-k expert indices [num_tokens, topk]
        gating_output: Gating logits [num_tokens, num_experts]
        renormalize: Whether to renormalize the top-k weights
        correction_bias: Per-expert bias correction [num_experts], must be float32 if provided
    """
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[-1]
    has_correction_bias = correction_bias is not None

    block_width_up = triton.next_power_of_2(num_experts)
    grid = (num_tokens,)

    topk_sigmoid_triton_kernel[grid](
        gating_output,
        topk_ids,
        topk_weights,
        renormalize,
        correction_bias,
        has_correction_bias,
        num_experts,
        num_tokens,
        K=topk,
        BLOCK_K=triton.next_power_of_2(topk),
        BLOCK_WIDTH_SIZE_UP=block_width_up,
    )
