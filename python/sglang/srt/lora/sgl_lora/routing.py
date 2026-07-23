"""Canonical virtual-expert routing for SGL LoRA MoE kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.virtual_experts import (
    _align_block_size_jit,
    _align_block_size_torch,
)


@dataclass(frozen=True)
class VirtualExpertRouting:
    """Aligned metadata over canonical ``(adapter, factor expert)`` IDs."""

    virtual_topk_ids: torch.Tensor
    sorted_pair_ids: torch.Tensor
    block_virtual_expert_ids: torch.Tensor
    num_pairs_post_padded: torch.Tensor
    num_virtual_experts: int
    block_size: int


@triton.jit
def _build_virtual_topk_ids_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    factor_map_ptr,
    virtual_topk_ids_ptr,
    num_pairs,
    factor_map_size,
    FACTOR_EXPERT_COUNT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    USE_FACTOR_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pair_mask = pair_ids < num_pairs
    token_ids = pair_ids // TOP_K

    adapter_ids = tl.load(
        token_lora_mapping_ptr + token_ids,
        mask=pair_mask,
        other=-1,
    ).to(tl.int64)
    routed_expert_ids = tl.load(
        topk_ids_ptr + pair_ids,
        mask=pair_mask,
        other=-1,
    ).to(tl.int64)
    if USE_FACTOR_MAP:
        in_map = (routed_expert_ids >= 0) & (
            routed_expert_ids < factor_map_size
        )
        factor_ids = tl.load(
            factor_map_ptr + routed_expert_ids,
            mask=pair_mask & in_map,
            other=-1,
        ).to(tl.int64)
    else:
        factor_ids = routed_expert_ids

    valid = (
        (adapter_ids >= 0)
        & (adapter_ids < MAX_LORAS)
        & (factor_ids >= 0)
        & (factor_ids < FACTOR_EXPERT_COUNT)
    )
    virtual_ids = adapter_ids * FACTOR_EXPERT_COUNT + factor_ids
    tl.store(
        virtual_topk_ids_ptr + pair_ids,
        tl.where(valid, virtual_ids, -1),
        mask=pair_mask,
    )


def _build_virtual_topk_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    factor_expert_count: int,
    max_loras: int,
    routed_expert_to_factor_id: torch.Tensor | None,
) -> torch.Tensor:
    virtual_topk_ids = torch.empty_like(topk_ids)
    if topk_ids.numel() == 0:
        return virtual_topk_ids

    if not topk_ids.is_cuda:
        adapter_valid = (token_lora_mapping >= 0) & (
            token_lora_mapping < max_loras
        )
        if routed_expert_to_factor_id is None:
            factor_ids = topk_ids
        elif routed_expert_to_factor_id.numel() == 0:
            factor_ids = torch.full_like(topk_ids, -1)
        else:
            in_map = (topk_ids >= 0) & (
                topk_ids < routed_expert_to_factor_id.numel()
            )
            safe_ids = topk_ids.clamp(
                min=0, max=routed_expert_to_factor_id.numel() - 1
            )
            factor_ids = torch.where(
                in_map,
                routed_expert_to_factor_id[safe_ids],
                -1,
            )
        factor_valid = (factor_ids >= 0) & (factor_ids < factor_expert_count)
        virtual_ids = (
            token_lora_mapping[:, None] * factor_expert_count + factor_ids
        )
        return torch.where(adapter_valid[:, None] & factor_valid, virtual_ids, -1)

    block_size = 1024
    factor_map = (
        topk_ids
        if routed_expert_to_factor_id is None
        else routed_expert_to_factor_id
    )
    _build_virtual_topk_ids_kernel[
        (triton.cdiv(topk_ids.numel(), block_size),)
    ](
        topk_ids,
        token_lora_mapping,
        factor_map,
        virtual_topk_ids,
        topk_ids.numel(),
        0 if routed_expert_to_factor_id is None else factor_map.numel(),
        FACTOR_EXPERT_COUNT=factor_expert_count,
        MAX_LORAS=max_loras,
        TOP_K=topk_ids.shape[1],
        USE_FACTOR_MAP=routed_expert_to_factor_id is not None,
        BLOCK_SIZE=block_size,
    )
    return virtual_topk_ids


def _routing_capacity(
    num_pairs: int,
    block_size: int,
    num_virtual_experts: int,
) -> int:
    if num_pairs == 0:
        return 0
    max_nonempty_buckets = min(num_pairs, num_virtual_experts + 1)
    upper_bound = num_pairs + max_nonempty_buckets * (block_size - 1)
    return triton.cdiv(triton.cdiv(upper_bound, block_size) * block_size, 4) * 4


def _align_aot(
    virtual_topk_ids: torch.Tensor,
    block_size: int,
    num_virtual_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.moe import moe_align_block_size

    flat_ids = virtual_topk_ids.reshape(-1)
    if flat_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.int32, device=flat_ids.device)
        return empty, empty, torch.zeros(1, dtype=torch.int32, device=flat_ids.device)

    capacity = _routing_capacity(
        flat_ids.numel(), block_size, num_virtual_experts
    )
    sorted_pair_ids = torch.empty(
        capacity, dtype=torch.int32, device=flat_ids.device
    )
    block_virtual_expert_ids = torch.empty(
        triton.cdiv(capacity, block_size),
        dtype=torch.int32,
        device=flat_ids.device,
    )
    num_pairs_post_padded = torch.empty(
        1, dtype=torch.int32, device=flat_ids.device
    )
    bucket_count = num_virtual_experts + 1
    cumsum = torch.empty(
        bucket_count + 1, dtype=torch.int32, device=flat_ids.device
    )
    moe_align_block_size(
        flat_ids,
        bucket_count,
        block_size,
        sorted_pair_ids,
        block_virtual_expert_ids,
        num_pairs_post_padded,
        cumsum,
        True,
    )
    return sorted_pair_ids, block_virtual_expert_ids, num_pairs_post_padded


def _align_virtual_topk_ids(
    virtual_topk_ids: torch.Tensor,
    block_size: int,
    num_virtual_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not virtual_topk_ids.is_cuda:
        return _align_block_size_torch(
            virtual_topk_ids, block_size, num_virtual_experts
        )
    if num_virtual_experts < 1024:
        return _align_aot(virtual_topk_ids, block_size, num_virtual_experts)
    if num_virtual_experts <= 8191:
        return _align_block_size_jit(
            virtual_topk_ids, block_size, num_virtual_experts
        )
    # This is a correctness fallback, not a performance policy. Large-domain
    # routing remains a measured optimization gap.
    return _align_block_size_torch(
        virtual_topk_ids, block_size, num_virtual_experts
    )


def build_virtual_expert_routing(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    factor_expert_count: int,
    max_loras: int,
    block_size: int,
    routed_expert_to_factor_id: torch.Tensor | None = None,
) -> VirtualExpertRouting:
    """Canonicalize and align token/expert pairs for flattened LoRA factors."""
    if topk_ids.ndim != 2 or token_lora_mapping.shape != (topk_ids.shape[0],):
        raise ValueError("expected topk_ids [T,K] and token_lora_mapping [T]")
    if min(factor_expert_count, max_loras, block_size) <= 0:
        raise ValueError(
            "factor count, adapter capacity, and block size must be positive"
        )

    num_virtual_experts = factor_expert_count * max_loras
    virtual_topk_ids = _build_virtual_topk_ids(
        topk_ids,
        token_lora_mapping,
        factor_expert_count,
        max_loras,
        routed_expert_to_factor_id,
    )
    sorted_pair_ids, block_virtual_expert_ids, num_pairs_post_padded = (
        _align_virtual_topk_ids(
            virtual_topk_ids,
            block_size,
            num_virtual_experts,
        )
    )
    return VirtualExpertRouting(
        virtual_topk_ids=virtual_topk_ids,
        sorted_pair_ids=sorted_pair_ids,
        block_virtual_expert_ids=block_virtual_expert_ids,
        num_pairs_post_padded=num_pairs_post_padded,
        num_virtual_experts=num_virtual_experts,
        block_size=block_size,
    )


__all__ = ["VirtualExpertRouting", "build_virtual_expert_routing"]
