# SPDX-License-Identifier: Apache-2.0
"""Clover block coreset selection in four kernel stages.

Compute per-token top-k scores, aggregate expert mass across each block,
select the top-p expert set, and mask router logits for Clover preview.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _float_descending_sort_key(values, valid):
    """Pack a descending float key and ascending expert id into int64."""
    expert_ids = tl.arange(0, 256)
    min_i32: tl.constexpr = -2147483648
    bits = values.to(tl.int32, bitcast=True)
    sign = bits >> 31
    key = tl.where(sign == 0, bits ^ -1, bits ^ min_i32)
    key = tl.where(valid, key, 0x7FFFFFFF)
    key64 = key.to(tl.int64) & 0x00000000FFFFFFFF
    return (key64 << 32) | expert_ids.to(tl.int64)


@triton.jit
def _clover_router_topk_scores_kernel(
    logits_ptr,
    generation_block_ids_ptr,
    eligible_ptr,
    topk_ids_ptr,
    topk_scores_ptr,
    logits_stride_t,
    logits_stride_e,
    TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_idx = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    valid = expert_offsets < NUM_EXPERTS
    logits = tl.load(
        logits_ptr + token_idx * logits_stride_t + expert_offsets * logits_stride_e,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    denominator = tl.sum(tl.exp(logits - row_max), axis=0)

    packed = _float_descending_sort_key(logits, valid)
    sorted_packed = tl.sort(packed, descending=False)
    sorted_ids = (sorted_packed & 0x00000000FFFFFFFF).to(tl.int32)
    topk_logits = tl.load(
        logits_ptr + token_idx * logits_stride_t + sorted_ids * logits_stride_e,
        mask=expert_offsets < TOP_K,
        other=-float("inf"),
    ).to(tl.float32)
    topk_scores = tl.exp(topk_logits - row_max) / denominator
    is_eligible = (tl.load(eligible_ptr + token_idx) != 0) & (
        tl.load(generation_block_ids_ptr + token_idx) >= 0
    )
    topk_scores = tl.where(is_eligible, topk_scores, 0.0)
    topk_offsets = token_idx * TOP_K + expert_offsets
    tl.store(
        topk_ids_ptr + topk_offsets,
        sorted_ids,
        mask=expert_offsets < TOP_K,
    )
    tl.store(
        topk_scores_ptr + topk_offsets,
        topk_scores,
        mask=expert_offsets < TOP_K,
    )


@triton.jit
def _clover_block_score_kernel(
    topk_ids_ptr,
    topk_scores_ptr,
    block_scores_ptr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    expert_offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    block_idx = tl.program_id(1)
    route_offsets = tl.arange(0, BLOCK_R)
    route_mask = route_offsets < BLOCK_SIZE * TOP_K
    route_base = block_idx * BLOCK_SIZE * TOP_K
    route_ids = tl.load(
        topk_ids_ptr + route_base + route_offsets, mask=route_mask, other=-1
    )
    route_scores = tl.load(
        topk_scores_ptr + route_base + route_offsets,
        mask=route_mask,
        other=0.0,
    ).to(tl.float32)
    matches = expert_offsets[:, None] == route_ids[None, :]
    scores = tl.sum(tl.where(matches, route_scores[None, :], 0.0), axis=1)
    tl.store(
        block_scores_ptr + block_idx * NUM_EXPERTS + expert_offsets,
        scores,
        mask=expert_offsets < NUM_EXPERTS,
    )


@triton.jit
def _clover_select_coreset_kernel(
    block_scores_ptr,
    block_allow_ptr,
    top_p,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    block_idx = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    valid = expert_offsets < NUM_EXPERTS
    scores = tl.load(
        block_scores_ptr + block_idx * NUM_EXPERTS + expert_offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    packed = _float_descending_sort_key(scores, valid)
    sorted_packed = tl.sort(packed, descending=False)
    sorted_ids = (sorted_packed & 0x00000000FFFFFFFF).to(tl.int32)
    sorted_scores = tl.load(
        block_scores_ptr + block_idx * NUM_EXPERTS + sorted_ids,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    total = tl.maximum(tl.sum(tl.where(valid, scores, 0.0), axis=0), 1.0e-9)
    cumulative = tl.cumsum(sorted_scores, axis=0)
    keep = (cumulative - sorted_scores) < top_p * total
    # At top_p=1 the mathematical coreset is exactly the union of experts
    # carrying positive top-k mass. Comparing two independently reduced FP32
    # totals can admit zero-mass tails (or drop the last positive lane).
    keep = tl.where(top_p >= 1.0, sorted_scores > 0.0, keep)
    keep = keep | (expert_offsets == 0)
    tl.store(
        block_allow_ptr + block_idx * NUM_EXPERTS + sorted_ids,
        keep,
        mask=valid,
    )


@triton.jit
def _clover_apply_coreset_kernel(
    logits_ptr,
    generation_block_ids_ptr,
    eligible_ptr,
    block_allow_ptr,
    output_ptr,
    logits_stride_t,
    logits_stride_e,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_idx = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    valid = expert_offsets < NUM_EXPERTS
    block_idx = token_idx // BLOCK_SIZE
    logits = tl.load(
        logits_ptr + token_idx * logits_stride_t + expert_offsets * logits_stride_e,
        mask=valid,
        other=0.0,
    )
    allow = tl.load(
        block_allow_ptr + block_idx * NUM_EXPERTS + expert_offsets,
        mask=valid,
        other=0,
    )
    is_eligible = (tl.load(eligible_ptr + token_idx) != 0) & (
        tl.load(generation_block_ids_ptr + token_idx) >= 0
    )
    output = tl.where(is_eligible & ~allow, -float("inf"), logits)
    tl.store(
        output_ptr + token_idx * NUM_EXPERTS + expert_offsets,
        output,
        mask=valid,
    )


def supports_clover_block_coreset_triton(
    router_logits: torch.Tensor, *, top_k: int, block_size: int
) -> bool:
    return (
        router_logits.is_cuda
        and router_logits.ndim == 2
        and router_logits.shape[1] == 256
        and block_size in (4, 8, 16, 32, 128)
        and router_logits.shape[0] > 0
        and router_logits.shape[0] % block_size == 0
        and top_k == 8
        and router_logits.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and torch.cuda.get_device_capability(router_logits.device)[0] in (9, 10)
    )


def _apply_clover_block_coreset_triton(
    router_logits: torch.Tensor,
    *,
    generation_block_ids: torch.Tensor,
    coreset_eligible: torch.Tensor,
    top_k: int,
    top_p: float,
    block_size: int,
) -> torch.Tensor:
    """Apply the fixed Clover block coreset without PyTorch sort/scatter ops."""
    if not supports_clover_block_coreset_triton(
        router_logits, top_k=top_k, block_size=block_size
    ):
        raise NotImplementedError(
            "Clover coreset Triton supports CUDA, block_size in {4, 8, 16, 32, 128}, "
            "top_k=8, and 256 experts"
        )
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"Clover coreset top-p must be in (0, 1], got {top_p}")
    num_tokens, num_experts = router_logits.shape
    if generation_block_ids.numel() != num_tokens:
        raise ValueError(
            f"{generation_block_ids.numel()=} does not match {num_tokens=}"
        )
    if coreset_eligible.numel() != num_tokens:
        raise ValueError(f"{coreset_eligible.numel()=} does not match {num_tokens=}")

    batch_size = num_tokens // block_size
    topk_ids = torch.empty(
        (num_tokens, top_k), dtype=torch.int32, device=router_logits.device
    )
    topk_scores = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=router_logits.device
    )
    block_scores = torch.empty(
        (batch_size, num_experts), dtype=torch.float32, device=router_logits.device
    )
    block_allow = torch.empty(
        (batch_size, num_experts), dtype=torch.bool, device=router_logits.device
    )
    output = torch.empty(
        router_logits.shape, dtype=router_logits.dtype, device=router_logits.device
    )

    # The latency-critical MRR=1 / block-32 preview replays this four-kernel
    # chain once per MoE layer inside a CUDA Graph.  On Blackwell, using four
    # warps consistently across the chain together with a 32-expert score tile
    # is faster than the generic 8-warp / 64-expert configuration.  Keep the
    # generic launch geometry for other batch and block shapes.
    single_block_32 = batch_size == 1 and block_size == 32
    num_warps = 4 if single_block_32 else 8
    score_block_e = 32 if single_block_32 or block_size == 128 else 64
    block_r = triton.next_power_of_2(block_size * top_k)

    _clover_router_topk_scores_kernel[(num_tokens,)](
        router_logits,
        generation_block_ids,
        coreset_eligible,
        topk_ids,
        topk_scores,
        router_logits.stride(0),
        router_logits.stride(1),
        TOP_K=top_k,
        NUM_EXPERTS=num_experts,
        BLOCK_E=256,
        num_warps=num_warps,
    )
    _clover_block_score_kernel[(triton.cdiv(num_experts, score_block_e), batch_size)](
        topk_ids,
        topk_scores,
        block_scores,
        TOP_K=top_k,
        BLOCK_SIZE=block_size,
        NUM_EXPERTS=num_experts,
        BLOCK_E=score_block_e,
        BLOCK_R=block_r,
        num_warps=num_warps,
    )
    _clover_select_coreset_kernel[(batch_size,)](
        block_scores,
        block_allow,
        float(top_p),
        NUM_EXPERTS=num_experts,
        BLOCK_E=256,
        num_warps=num_warps,
    )
    _clover_apply_coreset_kernel[(num_tokens,)](
        router_logits,
        generation_block_ids,
        coreset_eligible,
        block_allow,
        output,
        router_logits.stride(0),
        router_logits.stride(1),
        BLOCK_SIZE=block_size,
        NUM_EXPERTS=num_experts,
        BLOCK_E=256,
        num_warps=num_warps,
    )
    return output


def clover_block_coreset_triton(
    logits: torch.Tensor, block_size: int, top_k: int, top_p: float
) -> torch.Tensor:
    """Select experts for full Clover preview blocks; every row is eligible."""
    return _apply_clover_block_coreset_triton(
        logits,
        generation_block_ids=torch.zeros(
            logits.shape[0], device=logits.device, dtype=torch.int64
        ),
        coreset_eligible=torch.ones(
            logits.shape[0], device=logits.device, dtype=torch.bool
        ),
        top_k=top_k,
        top_p=top_p,
        block_size=block_size,
    )
