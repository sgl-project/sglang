"""Device tensor reference planner for a persistent, per-request sparse KV LRU."""

from typing import NamedTuple

import torch


class SparseLRUPlan(NamedTuple):
    miss_slots: torch.Tensor
    evicted_tokens: torch.Tensor
    next_order: torch.Tensor


def plan_sparse_lru(
    *,
    hit_mask: torch.Tensor,
    hit_slots: torch.Tensor,
    miss_mask: torch.Tensor,
    cached_tokens: torch.Tensor,
    lru_order: torch.Tensor,
) -> SparseLRUPlan:
    """Protect all hits before assigning unique misses to oldest unused slots.

    Shapes are [B, K] for selections and [B, C] for cache state, C >= K.
    Valid top-k IDs must be distinct within each request (the DSA TopK contract).
    Entries accessed in one decode round share recency; their tie order follows
    the previous order for hits and top-k order for misses. All shapes are static;
    no tensor values are read on the host. This reference planner uses a sort and
    should be replaced by native LIM metadata when that ABI is available.
    """
    batch_size, capacity = cached_tokens.shape
    if capacity < hit_mask.shape[1]:
        raise ValueError("Sparse LRU capacity must cover the complete top-k set.")
    selected = torch.zeros(
        (batch_size, capacity + 1), dtype=torch.int32, device=hit_mask.device
    )
    safe_hit_slots = torch.where(hit_mask, hit_slots.long(), capacity)
    # All non-hits target a sentinel column; real hit slots are unique.
    selected.scatter_(1, safe_hit_slots, 1)
    hit_in_lru_order = selected.gather(1, lru_order)
    # Binary keys are exactly representable in FP32. Integer ArgSort dispatches
    # to AiCPU on Ascend; keep this reference planner on AiCore.
    permutation = torch.argsort(hit_in_lru_order.float(), dim=1, stable=True)
    evictable_first = lru_order.gather(1, permutation)
    miss_rank = (miss_mask.long().cumsum(1) - 1).clamp(min=0)
    miss_slots = evictable_first.gather(1, miss_rank)
    evicted_tokens = cached_tokens.gather(1, miss_slots)
    # Move assigned victims behind all retained entries and current hits.
    offsets = torch.arange(capacity, device=hit_mask.device).unsqueeze(0)
    rotation = (offsets + miss_mask.long().sum(1, keepdim=True)) % capacity
    next_order = evictable_first.gather(1, rotation)
    return SparseLRUPlan(miss_slots, evicted_tokens, next_order)
