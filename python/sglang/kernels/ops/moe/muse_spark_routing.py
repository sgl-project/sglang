"""
Fused Triton kernel for MuseSpark MoE custom routing.

This module provides a single fused Triton kernel that replaces the multi-kernel
implementation of MuseSpark routing. The fused kernel performs all routing
operations in a single pass:
1. Optional multiply by router_output_multiplier
2. Sigmoid or softmax normalization
3. Add bias for selection scoring
4. Iterative top-k selection
5. Gather normalized scores for selected experts
6. Normalize by sum (+ eps) and scale
7. Return results

The kernel is CUDA graph compatible by accepting pre-allocated output buffers.

Not ``moe.moe_fused_gate``: that router has no logit multiplier or EP
local-expert map, adds the bias before softmax, zeroes padded-row weights, and
rounds weights as ``score / (sum + eps) * scale`` instead of
``score * (scale / (sum + eps))``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.spec import CapabilityRequirement, KernelBackend
from sglang.srt.environ import envs
from sglang.srt.utils import fast_topk
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _compute_normalized_scores(
    gating_scores,
    expert_mask,
    NORMALIZATION_MODE: tl.constexpr,
):
    """Compute normalized scores (sigmoid or softmax)."""
    if NORMALIZATION_MODE == 0:
        # Sigmoid normalization
        return tl.sigmoid(gating_scores)
    else:
        # Softmax normalization with numerical stability
        max_score = tl.max(tl.where(expert_mask, gating_scores, -float("inf")), axis=0)
        exp_scores = tl.exp(gating_scores - max_score)
        exp_scores = tl.where(expert_mask, exp_scores, 0.0)
        sum_exp = tl.sum(exp_scores, axis=0)
        return exp_scores / sum_exp


@triton.jit
def _select_expert_and_score(
    scores_for_topk,
    normalized_scores,
    selected_mask,
    expert_mask,
    expert_offs,
):
    """Select the top expert and return (max_idx, selected_score, updated_mask)."""
    # Mask out already selected experts
    masked_scores = tl.where(selected_mask, -float("inf"), scores_for_topk)
    masked_scores = tl.where(expert_mask, masked_scores, -float("inf"))

    # Find the index of the maximum (argmax)
    max_idx = tl.argmax(masked_scores, axis=0)

    # Mark this expert as selected
    is_selected = expert_offs == max_idx
    new_selected_mask = selected_mask | is_selected

    # Get the normalized score for this expert
    selected_score = tl.sum(tl.where(is_selected, normalized_scores, 0.0), axis=0)

    return max_idx, selected_score, new_selected_mask


# =============================================================================
# Specialized unrolled kernel for topk=8
# =============================================================================


@triton.jit
def fused_muse_spark_routing_kernel_topk8(
    gating_output_ptr,
    bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    packed_ptr,
    local_expert_mapping_ptr,
    num_valid_tokens_ptr,
    num_tokens,
    num_experts,
    stride_gm,
    stride_gn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    stride_pm,
    router_output_multiplier,
    route_scale,
    division_eps,
    NUM_EXPERTS_PADDED: tl.constexpr,
    NORMALIZATION_MODE: tl.constexpr,
    HAS_MULTIPLIER: tl.constexpr,
    HAS_EPS: tl.constexpr,
    USE_GDC: tl.constexpr,
    HAS_LOCAL_EXPERT_MAPPING: tl.constexpr,
    HAS_NUM_VALID: tl.constexpr,
    HAS_PACKED: tl.constexpr,
) -> None:
    """Optimized kernel for topk=8 (fully unrolled, fused sum accumulation)."""
    token_id = tl.program_id(0)
    if token_id >= num_tokens:
        return

    # Padded-row short-circuit: rows >= the (device-scalar) real token count are
    # padding. Emit a sentinel expert id of -1 so the MoE backend skips them
    # (topk_id==-1 matches no expert). See num_token_non_padded plumbing.
    if HAS_NUM_VALID:
        n_valid = tl.load(num_valid_tokens_ptr)
        if token_id >= n_valid:
            pad_k = tl.arange(0, 8)
            tl.store(
                topk_ids_ptr + token_id * stride_im + pad_k * stride_ik,
                tl.full((8,), -1, tl.int32),
            )
            if HAS_PACKED:
                tl.store(
                    packed_ptr + token_id * stride_pm + pad_k,
                    tl.full((8,), -1 << 16, tl.int32),
                )
            return

    expert_offs = tl.arange(0, NUM_EXPERTS_PADDED)
    expert_mask = expert_offs < num_experts

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    gating_ptr = gating_output_ptr + token_id * stride_gm
    gating_scores = tl.load(
        gating_ptr + expert_offs * stride_gn, mask=expert_mask, other=-float("inf")
    )

    if HAS_MULTIPLIER:
        gating_scores = gating_scores * router_output_multiplier

    normalized_scores = _compute_normalized_scores(
        gating_scores, expert_mask, NORMALIZATION_MODE
    )

    bias_values = tl.load(bias_ptr + expert_offs, mask=expert_mask, other=0.0)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

    scores_for_topk = normalized_scores + bias_values
    selected_mask = tl.zeros([NUM_EXPERTS_PADDED], dtype=tl.int1)

    # Unrolled selections 0-7
    max_idx_0, score_0, selected_mask = _select_expert_and_score(
        scores_for_topk, normalized_scores, selected_mask, expert_mask, expert_offs
    )
    max_idx_1, score_1, selected_mask = _select_expert_and_score(
        scores_for_topk, normalized_scores, selected_mask, expert_mask, expert_offs
    )
    max_idx_2, score_2, selected_mask = _select_expert_and_score(
        scores_for_topk, normalized_scores, selected_mask, expert_mask, expert_offs
    )
    max_idx_3, score_3, selected_mask = _select_expert_and_score(
        scores_for_topk, normalized_scores, selected_mask, expert_mask, expert_offs
    )
    max_idx_4, score_4, selected_mask = _select_expert_and_score(
        scores_for_topk, normalized_scores, selected_mask, expert_mask, expert_offs
    )
    max_idx_5, score_5, selected_mask = _select_expert_and_score(
        scores_for_topk, normalized_scores, selected_mask, expert_mask, expert_offs
    )
    max_idx_6, score_6, selected_mask = _select_expert_and_score(
        scores_for_topk, normalized_scores, selected_mask, expert_mask, expert_offs
    )
    max_idx_7, score_7, _ = _select_expert_and_score(
        scores_for_topk, normalized_scores, selected_mask, expert_mask, expert_offs
    )

    # Fused sum and normalization
    score_sum = (
        score_0 + score_1 + score_2 + score_3 + score_4 + score_5 + score_6 + score_7
    )
    if HAS_EPS:
        score_sum = score_sum + division_eps

    inv_sum_scaled = route_scale / score_sum

    # Apply local expert mapping if provided (EP global→local ID remapping)
    if HAS_LOCAL_EXPERT_MAPPING:
        max_idx_0 = tl.load(local_expert_mapping_ptr + max_idx_0)
        max_idx_1 = tl.load(local_expert_mapping_ptr + max_idx_1)
        max_idx_2 = tl.load(local_expert_mapping_ptr + max_idx_2)
        max_idx_3 = tl.load(local_expert_mapping_ptr + max_idx_3)
        max_idx_4 = tl.load(local_expert_mapping_ptr + max_idx_4)
        max_idx_5 = tl.load(local_expert_mapping_ptr + max_idx_5)
        max_idx_6 = tl.load(local_expert_mapping_ptr + max_idx_6)
        max_idx_7 = tl.load(local_expert_mapping_ptr + max_idx_7)

    # Store outputs
    out_weights_ptr = topk_weights_ptr + token_id * stride_wm
    out_ids_ptr = topk_ids_ptr + token_id * stride_im

    tl.store(out_ids_ptr + 0 * stride_ik, max_idx_0.to(tl.int32))
    tl.store(out_ids_ptr + 1 * stride_ik, max_idx_1.to(tl.int32))
    tl.store(out_ids_ptr + 2 * stride_ik, max_idx_2.to(tl.int32))
    tl.store(out_ids_ptr + 3 * stride_ik, max_idx_3.to(tl.int32))
    tl.store(out_ids_ptr + 4 * stride_ik, max_idx_4.to(tl.int32))
    tl.store(out_ids_ptr + 5 * stride_ik, max_idx_5.to(tl.int32))
    tl.store(out_ids_ptr + 6 * stride_ik, max_idx_6.to(tl.int32))
    tl.store(out_ids_ptr + 7 * stride_ik, max_idx_7.to(tl.int32))

    tl.store(out_weights_ptr + 0 * stride_wk, score_0 * inv_sum_scaled)
    tl.store(out_weights_ptr + 1 * stride_wk, score_1 * inv_sum_scaled)
    tl.store(out_weights_ptr + 2 * stride_wk, score_2 * inv_sum_scaled)
    tl.store(out_weights_ptr + 3 * stride_wk, score_3 * inv_sum_scaled)
    tl.store(out_weights_ptr + 4 * stride_wk, score_4 * inv_sum_scaled)
    tl.store(out_weights_ptr + 5 * stride_wk, score_5 * inv_sum_scaled)
    tl.store(out_weights_ptr + 6 * stride_wk, score_6 * inv_sum_scaled)
    tl.store(out_weights_ptr + 7 * stride_wk, score_7 * inv_sum_scaled)

    if HAS_PACKED:
        out_packed_ptr = packed_ptr + token_id * stride_pm
        tl.store(
            out_packed_ptr + 0, _pack_id_weight(max_idx_0, score_0 * inv_sum_scaled)
        )
        tl.store(
            out_packed_ptr + 1, _pack_id_weight(max_idx_1, score_1 * inv_sum_scaled)
        )
        tl.store(
            out_packed_ptr + 2, _pack_id_weight(max_idx_2, score_2 * inv_sum_scaled)
        )
        tl.store(
            out_packed_ptr + 3, _pack_id_weight(max_idx_3, score_3 * inv_sum_scaled)
        )
        tl.store(
            out_packed_ptr + 4, _pack_id_weight(max_idx_4, score_4 * inv_sum_scaled)
        )
        tl.store(
            out_packed_ptr + 5, _pack_id_weight(max_idx_5, score_5 * inv_sum_scaled)
        )
        tl.store(
            out_packed_ptr + 6, _pack_id_weight(max_idx_6, score_6 * inv_sum_scaled)
        )
        tl.store(
            out_packed_ptr + 7, _pack_id_weight(max_idx_7, score_7 * inv_sum_scaled)
        )


@triton.jit
def _pack_id_weight(expert_id, weight):
    w_i16 = weight.to(tl.bfloat16).to(tl.int16, bitcast=True)
    return (expert_id.to(tl.int32) << 16) | (w_i16.to(tl.int32) & 0xFFFF)


# =============================================================================
# Generic kernel (fallback for non-topk8)
# =============================================================================


@triton.jit
def fused_muse_spark_routing_kernel(
    # Input pointers
    gating_output_ptr,
    bias_ptr,
    # Output pointers
    topk_weights_ptr,
    topk_ids_ptr,
    # Local expert mapping pointer (optional, controlled by HAS_LOCAL_EXPERT_MAPPING)
    local_expert_mapping_ptr,
    # Device-scalar real (non-padded) token count (optional, HAS_NUM_VALID)
    num_valid_tokens_ptr,
    # Dimensions
    num_tokens,
    num_experts,
    # Strides
    stride_gm,
    stride_gn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    # Scalar parameters
    router_output_multiplier,
    route_scale,
    division_eps,
    # Compile-time constants
    TOPK: tl.constexpr,
    NUM_EXPERTS_PADDED: tl.constexpr,
    NORMALIZATION_MODE: tl.constexpr,  # 0=sigmoid, 1=softmax
    HAS_MULTIPLIER: tl.constexpr,
    HAS_EPS: tl.constexpr,
    USE_GDC: tl.constexpr,
    HAS_LOCAL_EXPERT_MAPPING: tl.constexpr,
    HAS_NUM_VALID: tl.constexpr,
) -> None:
    """
    Fused kernel for MuseSpark MoE routing.

    Each program handles one token (row). The algorithm:
    1. Load all expert scores for this token
    2. Apply optional multiplier
    3. Apply sigmoid or softmax normalization
    4. Add bias for top-k selection
    5. Iterative top-k: loop TOPK times, finding max and masking
    6. Gather normalized scores for selected experts
    7. Normalize by sum (+ eps) and scale
    8. Store results
    """
    # One program per token
    token_id = tl.program_id(0)

    if token_id >= num_tokens:
        return

    # Padded-row short-circuit (see topk8 kernel): emit topk_id=-1 for rows >=
    # the real token count so the MoE backend skips them.
    if HAS_NUM_VALID:
        n_valid = tl.load(num_valid_tokens_ptr)
        if token_id >= n_valid:
            pad_k = tl.arange(0, TOPK)
            tl.store(
                topk_ids_ptr + token_id * stride_im + pad_k * stride_ik,
                tl.full((TOPK,), -1, tl.int32),
            )
            return

    # Create offsets for experts dimension
    expert_offs = tl.arange(0, NUM_EXPERTS_PADDED)
    expert_mask = expert_offs < num_experts

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    # Load gating output for this token [num_experts]
    gating_ptr = gating_output_ptr + token_id * stride_gm
    gating_scores = tl.load(
        gating_ptr + expert_offs * stride_gn, mask=expert_mask, other=-float("inf")
    )

    # Step 1: Apply optional multiplier
    if HAS_MULTIPLIER:
        gating_scores = gating_scores * router_output_multiplier

    # Step 2: Apply normalization (sigmoid or softmax)
    if NORMALIZATION_MODE == 0:
        # Sigmoid normalization
        normalized_scores = tl.sigmoid(gating_scores)
    else:
        # Softmax normalization
        max_score = tl.max(tl.where(expert_mask, gating_scores, -float("inf")), axis=0)
        exp_scores = tl.exp(gating_scores - max_score)
        exp_scores = tl.where(expert_mask, exp_scores, 0.0)
        sum_exp = tl.sum(exp_scores, axis=0)
        normalized_scores = exp_scores / sum_exp

    # Step 3: Load bias and add for selection scoring
    bias_values = tl.load(bias_ptr + expert_offs, mask=expert_mask, other=0.0)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

    scores_for_topk = normalized_scores + bias_values

    # Step 4 & 5: Iterative top-k selection
    selected_mask = tl.zeros([NUM_EXPERTS_PADDED], dtype=tl.int1)

    # Output pointers for this token
    out_weights_ptr = topk_weights_ptr + token_id * stride_wm
    out_ids_ptr = topk_ids_ptr + token_id * stride_im

    # Accumulate scores in registers to avoid redundant memory round-trips
    score_sum = 0.0

    # Iterative top-k: find max k times, accumulating sum in registers
    for k in range(TOPK):
        # Mask out already selected experts
        masked_scores = tl.where(selected_mask, -float("inf"), scores_for_topk)
        masked_scores = tl.where(expert_mask, masked_scores, -float("inf"))

        # Find the index of the maximum (argmax)
        max_idx = tl.argmax(masked_scores, axis=0)

        # Mark this expert as selected
        selected_mask = selected_mask | (expert_offs == max_idx)

        # Get the normalized score (not the biased score) for this expert
        selected_score = tl.sum(
            tl.where(expert_offs == max_idx, normalized_scores, 0.0), axis=0
        )

        # Accumulate sum in register
        score_sum = score_sum + selected_score

        # Apply local expert mapping if provided
        if HAS_LOCAL_EXPERT_MAPPING:
            max_idx = tl.load(local_expert_mapping_ptr + max_idx)

        # Store the selected expert ID
        tl.store(out_ids_ptr + k * stride_ik, max_idx.to(tl.int32))

        # Store unnormalized score (will normalize after loop)
        tl.store(out_weights_ptr + k * stride_wk, selected_score)

    # Add epsilon if needed
    if HAS_EPS:
        score_sum = score_sum + division_eps

    # Normalize and store - only one pass needed now
    for k in range(TOPK):
        score_k = tl.load(out_weights_ptr + k * stride_wk)
        normalized_weight = (score_k / score_sum) * route_scale
        tl.store(out_weights_ptr + k * stride_wk, normalized_weight)


# =============================================================================
# Python wrapper
# =============================================================================


def _get_padded_num_experts(num_experts: int) -> int:
    """Get the next power of 2 >= num_experts (minimum 8) for efficient Triton execution."""
    return 1 << max(3, (num_experts - 1).bit_length())


def _is_pow2(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


@register_custom_op(
    op_name="fused_muse_spark_routing",
    mutates_args=["topk_weights", "topk_ids", "packed_topk_ids"],
)
def fused_muse_spark_routing_op(
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    packed_topk_ids: Optional[torch.Tensor],
    topk: int,
    route_scale: float,
    multiplier_val: float,
    eps_val: float,
    normalization_mode: int,
    has_multiplier: bool,
    has_eps: bool,
    launch_pdl: bool = False,
    local_expert_mapping: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> None:
    """Custom op wrapper for the fused MuseSpark routing Triton kernel.

    Registered as torch.ops.sglang.fused_muse_spark_routing so that torch.compile
    treats it as an opaque node (no tracing into Triton kernel launches).

    ``num_valid_tokens`` is an optional device int32 scalar (e.g.
    ``forward_batch.num_token_non_padded``): rows >= it are emitted as
    ``topk_id=-1`` so the MoE backend skips padded rows. Passing a device
    tensor (not a Python int) keeps it valid under CUDA-graph replay.
    """
    num_tokens, num_experts = gating_output.shape
    num_experts_padded = _get_padded_num_experts(num_experts)
    bias_f32 = bias.float() if bias.dtype != torch.float32 else bias
    has_local_expert_mapping = local_expert_mapping is not None
    has_num_valid = num_valid_tokens is not None
    has_packed = packed_topk_ids is not None

    grid = (num_tokens,)

    if topk == 8:
        fused_muse_spark_routing_kernel_topk8[grid](
            gating_output,
            bias_f32,
            topk_weights,
            topk_ids,
            packed_topk_ids,
            local_expert_mapping,
            num_valid_tokens,
            num_tokens,
            num_experts,
            gating_output.stride(0),
            gating_output.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            topk_ids.stride(0),
            topk_ids.stride(1),
            packed_topk_ids.stride(0) if has_packed else 0,
            multiplier_val,
            route_scale,
            eps_val,
            NUM_EXPERTS_PADDED=num_experts_padded,
            NORMALIZATION_MODE=normalization_mode,
            HAS_MULTIPLIER=has_multiplier,
            HAS_EPS=has_eps,
            USE_GDC=launch_pdl,
            HAS_LOCAL_EXPERT_MAPPING=has_local_expert_mapping,
            HAS_NUM_VALID=has_num_valid,
            HAS_PACKED=has_packed,
            launch_pdl=launch_pdl,
        )
    else:
        fused_muse_spark_routing_kernel[grid](
            gating_output,
            bias_f32,
            topk_weights,
            topk_ids,
            local_expert_mapping,
            num_valid_tokens,
            num_tokens,
            num_experts,
            gating_output.stride(0),
            gating_output.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            topk_ids.stride(0),
            topk_ids.stride(1),
            multiplier_val,
            route_scale,
            eps_val,
            TOPK=topk,
            NUM_EXPERTS_PADDED=num_experts_padded,
            NORMALIZATION_MODE=normalization_mode,
            HAS_MULTIPLIER=has_multiplier,
            HAS_EPS=has_eps,
            USE_GDC=launch_pdl,
            HAS_LOCAL_EXPERT_MAPPING=has_local_expert_mapping,
            HAS_NUM_VALID=has_num_valid,
            launch_pdl=launch_pdl,
        )
        if has_packed:
            packed_topk_ids.copy_(pack_topk_ids_native(topk_ids, topk_weights))


def pack_topk_ids_native(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """``(id << 16) | bf16_bits(weight)``; padded rows (id -1) carry weight 0."""
    weight_bits = (
        topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    )
    weight_bits = torch.where(topk_ids < 0, torch.zeros_like(weight_bits), weight_bits)
    return (topk_ids.to(torch.int32) << 16) | weight_bits


def fused_muse_spark_routing(
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    router_output_multiplier: Optional[float] = None,
    router_score_normalization: str = "sigmoid",
    router_score_division_eps: Optional[float] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    local_expert_mapping: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
    want_packed: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Fused MuseSpark routing function using a single Triton kernel.

    Pre-allocates output buffers and dispatches to the registered custom op
    (torch.ops.sglang.fused_muse_spark_routing) for piecewise CUDA graph
    compatibility.

    Args:
        gating_output: Router logits [num_tokens, num_experts]
        bias: Expert bias for selection [num_experts]
        topk: Number of experts to select per token
        route_scale: Scaling factor for final weights
        router_output_multiplier: Optional multiplier for gating output
        router_score_normalization: "sigmoid" or "softmax"
        router_score_division_eps: Optional epsilon for division stability
        topk_weights: Pre-allocated output buffer for weights [num_tokens, topk]
        topk_ids: Pre-allocated output buffer for expert IDs [num_tokens, topk]
        local_expert_mapping: Optional mapping table [num_experts] of int32,
            maps global expert IDs to local expert IDs for EP.
        num_valid_tokens: Optional device int32 scalar; rows >= it are emitted as
            topk_id=-1 so the MoE backend skips padded rows.

        want_packed: Also return the TRT-LLM packed ids,
            ``(expert_id << 16) | bf16_bits(weight)`` as int32 [num_tokens, topk].

    Returns:
        Tuple of (topk_weights, topk_ids, packed_topk_ids or None) tensors
    """
    assert gating_output.is_contiguous(), "gating_output must be contiguous"
    assert gating_output.dim() == 2, "gating_output must be 2D"
    assert bias.is_contiguous(), "bias must be contiguous"

    num_tokens, num_experts = gating_output.shape
    assert bias.shape[0] == num_experts, "bias size must match num_experts"

    # Validate normalization mode
    if router_score_normalization == "sigmoid":
        normalization_mode = 0
    elif router_score_normalization == "softmax":
        normalization_mode = 1
    else:
        raise ValueError(
            f"Invalid router_score_normalization: {router_score_normalization}. "
            "Expected 'sigmoid' or 'softmax'."
        )

    # Allocate output buffers if not provided
    if topk_weights is None:
        topk_weights = torch.empty(
            (num_tokens, topk),
            dtype=torch.float32,
            device=gating_output.device,
        )
    if topk_ids is None:
        topk_ids = torch.empty(
            (num_tokens, topk),
            dtype=torch.int32,
            device=gating_output.device,
        )

    # Ensure output buffers have correct shape
    assert topk_weights.shape[0] >= num_tokens, "topk_weights buffer too small"
    assert topk_weights.shape[1] >= topk, "topk_weights buffer too small"
    assert topk_ids.shape[0] >= num_tokens, "topk_ids buffer too small"
    assert topk_ids.shape[1] >= topk, "topk_ids buffer too small"

    # Use views if buffers are larger than needed
    topk_weights_view = topk_weights[:num_tokens, :topk]
    topk_ids_view = topk_ids[:num_tokens, :topk]
    packed_topk_ids = (
        torch.empty((num_tokens, topk), dtype=torch.int32, device=gating_output.device)
        if want_packed
        else None
    )

    # Convert Optional params to concrete values for custom op compatibility
    multiplier_val = (
        router_output_multiplier if router_output_multiplier is not None else 1.0
    )
    eps_val = (
        router_score_division_eps if router_score_division_eps is not None else 0.0
    )

    # Dispatch to registered custom op (opaque to torch.compile)
    fused_muse_spark_routing_op(
        gating_output,
        bias,
        topk_weights_view,
        topk_ids_view,
        packed_topk_ids,
        topk,
        route_scale,
        multiplier_val,
        eps_val,
        normalization_mode,
        router_output_multiplier is not None,
        router_score_division_eps is not None,
        launch_pdl=is_arch_support_pdl(),
        local_expert_mapping=local_expert_mapping,
        num_valid_tokens=num_valid_tokens,
    )

    return topk_weights_view, topk_ids_view, packed_topk_ids


# =============================================================================
# BaseFusedOp wrapper
# =============================================================================


class FusedMuseSparkRouting(BaseFusedOp):
    """Fused MuseSpark MoE routing with multi-backend dispatch.

    Routing logic:
    1. Optional metap scaling on router logits
    2. Sigmoid normalization
    3. Add expert_bias for top-k selection only
    4. Gather unbiased sigmoid scores for selected experts
    5. Renormalize: scores / (sum(scores) + eps) * route_scale
    """

    op = "moe.muse_spark_routing"
    priority = (KernelBackend.JIT, KernelBackend.TRITON)
    capabilities = {
        KernelBackend.JIT: {CapabilityRequirement.CUDA},
        KernelBackend.TRITON: {
            CapabilityRequirement.CUDA,
        },
    }

    def __init__(
        self,
        expert_bias: torch.Tensor,
        route_scale: float,
        router_output_multiplier: Optional[float] = None,
        router_score_division_eps: Optional[float] = None,
        router_score_normalization: str = "sigmoid",
    ):
        super().__init__()
        self.expert_bias = expert_bias
        self.route_scale = route_scale
        self.router_output_multiplier = router_output_multiplier
        self.router_score_division_eps = router_score_division_eps
        self.router_score_normalization = router_score_normalization
        self.local_expert_mapping: Optional[torch.Tensor] = None
        # SoL activated-expert capture. When set (by the owning MuseSparkMoE on decode-class
        # forwards), the fused router writes its topk_ids into this buffer so the SoL reporter
        # can count activated experts -- the router's own write IS the capture, no extra op.
        # None => normal transient topk_ids (e.g. prefill). The capacity check in forward_triton
        # is a safety net.
        self.sol_topk_out: Optional[torch.Tensor] = None
        # Optional device int32 scalar = real (non-padded) token count. When set,
        # routing emits topk_id=-1 for padded rows so the MoE backend skips them.
        # Set per-forward by MuseSparkMoE.forward; reset to None otherwise so
        # backends that don't expect -1 expert ids never see them.
        self.num_valid_tokens: Optional[torch.Tensor] = None

    def set_local_expert_mapping(self, mapping: Optional[torch.Tensor]) -> None:
        """Set the local expert mapping table for EP global→local remapping."""
        self.local_expert_mapping = mapping

    def set_num_valid_tokens(self, num_valid_tokens: Optional[torch.Tensor]) -> None:
        """Set the device-scalar real token count for padded-row masking.

        Pass ``forward_batch.num_token_non_padded`` (the local per-rank
        non-padded count). ``None`` disables masking (all rows treated real).
        """
        self.num_valid_tokens = num_valid_tokens

    def backend_eligible(
        self,
        backend: KernelBackend,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        want_packed: bool = False,
    ) -> bool:
        if not super().backend_eligible(
            backend, hidden_states, gating_output, topk, renormalize, want_packed
        ):
            return False
        if backend is not KernelBackend.JIT:
            return True
        # 128-bit row loads need num_experts % 4; the rest are static_asserts
        # in routing.cuh that would otherwise fail at compile time.
        return (
            1 <= topk <= 32
            and gating_output.shape[-1] % 4 == 0
            and self.router_score_normalization in ("sigmoid", "softmax")
        )

    def forward_jit(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        want_packed: bool = False,
    ):
        """Fused JIT CUDA kernel implementation."""
        assert renormalize, "MuseSpark routing requires renormalize=True"
        from sglang.kernels.ops.moe.muse_spark_routing_jit import (
            fused_muse_spark_routing_jit,
        )

        # See forward_triton: the SoL buffer is written by the router itself.
        cap = self.sol_topk_out
        if cap is not None and gating_output.shape[0] > cap.shape[0]:
            cap = None
        gating = gating_output.float().contiguous()
        packed = (
            torch.empty(
                (gating.shape[0], topk), dtype=torch.int32, device=gating.device
            )
            if want_packed
            else None
        )
        topk_weights, topk_ids, packed = fused_muse_spark_routing_jit(
            gating,
            self.expert_bias,
            topk,
            self.route_scale,
            router_output_multiplier=self.router_output_multiplier,
            router_score_normalization=self.router_score_normalization,
            router_score_division_eps=self.router_score_division_eps,
            topk_ids=cap,
            local_expert_mapping=self.local_expert_mapping,
            num_valid_tokens=self.num_valid_tokens,
            packed_topk_ids=packed,
        )
        if want_packed:
            return topk_weights, topk_ids, packed
        return topk_weights, topk_ids

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        want_packed: bool = False,
    ):
        """PyTorch reference implementation."""
        assert renormalize, "MuseSpark routing requires renormalize=True"
        if self.router_output_multiplier is not None:
            gating_output = gating_output * self.router_output_multiplier

        scores = torch.sigmoid(gating_output.float())

        scores_for_topk = scores + self.expert_bias.to(scores.device)
        _, topk_indices = fast_topk(scores_for_topk, topk, dim=-1)
        topk_indices = topk_indices.to(torch.int32)

        topk_scores = scores.gather(1, topk_indices.long())

        return self._finalize(topk_scores, topk_indices, gating_output, want_packed)

    def _finalize(
        self,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
        gating_output: torch.Tensor,
        want_packed: bool,
    ):
        """Renormalize, remap to local experts, mask padding, optionally pack.

        Shared by forward_native and forward_cpu so the two cannot drift; the
        backends differ only in how they produce the raw scores and indices.
        """
        if envs.SGLANG_USE_CUSTOM_ROUTER_SUM.get():
            from sglang.kernels.ops.elementwise.row_sum import row_sum

            score_sum = row_sum(topk_scores, dim=-1, keepdim=True)
        else:
            score_sum = topk_scores.sum(dim=-1, keepdim=True)
        if self.router_score_division_eps is not None:
            topk_scores = (
                topk_scores
                / (score_sum + self.router_score_division_eps)
                * self.route_scale
            )
        else:
            topk_scores = topk_scores / score_sum * self.route_scale

        if self.local_expert_mapping is not None:
            topk_indices = self.local_expert_mapping[topk_indices.long()]

        # Padded rows (>= real token count) get topk_id=-1 so the MoE backend
        # skips them.
        if self.num_valid_tokens is not None:
            num_tokens = gating_output.shape[0]
            row = torch.arange(num_tokens, device=topk_indices.device).unsqueeze(1)
            pad_mask = row >= self.num_valid_tokens.to(row.device)
            topk_indices = torch.where(
                pad_mask, torch.full_like(topk_indices, -1), topk_indices
            )

        if want_packed:
            return (
                topk_scores,
                topk_indices,
                pack_topk_ids_native(topk_indices, topk_scores),
            )
        return topk_scores, topk_indices

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        want_packed: bool = False,
    ):
        """sgl-kernel sigmoid + biased top-k; the tail stays shared.

        The kernel's own renormalization has no division epsilon, so it runs
        with renormalize=False and ``_finalize`` does the scaling.
        """
        assert renormalize, "MuseSpark routing requires renormalize=True"
        if self.router_output_multiplier is not None:
            gating_output = gating_output * self.router_output_multiplier

        gating_output = gating_output.contiguous()
        topk_scores, topk_indices = torch.ops.sgl_kernel.topk_sigmoid_cpu(
            gating_output,
            gating_output,
            topk,
            False,
            self.expert_bias.float(),
        )
        return self._finalize(
            topk_scores, topk_indices.to(torch.int32), gating_output, want_packed
        )

    def forward_triton(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        want_packed: bool = False,
    ):
        """Fused Triton kernel implementation."""
        assert renormalize, "MuseSpark routing requires renormalize=True"
        # Capture routing into the SoL buffer (the routing's own write is the capture
        # -- no copy). The decode-vs-prefill gate is upstream: MuseSparkMoE.forward only
        # sets sol_topk_out on decode-class forwards (None for prefill). This
        # remaining shape check is just a buffer-capacity safety net.
        cap = self.sol_topk_out
        if cap is not None and gating_output.shape[0] > cap.shape[0]:
            cap = None
        topk_weights, topk_ids, packed = fused_muse_spark_routing(
            gating_output.float().contiguous(),
            self.expert_bias,
            topk,
            self.route_scale,
            router_output_multiplier=self.router_output_multiplier,
            router_score_normalization=self.router_score_normalization,
            router_score_division_eps=self.router_score_division_eps,
            topk_ids=cap,
            local_expert_mapping=self.local_expert_mapping,
            num_valid_tokens=self.num_valid_tokens,
            want_packed=want_packed,
        )
        if want_packed:
            return topk_weights, topk_ids, packed
        return topk_weights, topk_ids
