from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, cast

import numpy as np
import torch

from sglang.kernels.ops.memory.common import (
    _get_last_loc_safe_kernel as _get_last_loc_safe_kernel,
)
from sglang.kernels.ops.memory.common import get_last_loc_kernel as get_last_loc_kernel
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.hicache_storage import PoolTransfer
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.runtime_context import get_serving, get_spec
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
# Lazy mode: 1 + 1 slots (1 ping-pong + 1 running), second ping-pong allocated on demand at boundary.
MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY = 2
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


class RetractionBackup(NamedTuple):
    cpu_tensors: Any = None
    host_indices: Optional[torch.Tensor] = None
    pool_transfers: Optional[list[PoolTransfer]] = None
    # Set when the KV pool leaves the recurrent state to the caller.
    mamba_cpu: Any = None


def kv_to_page_indices(kv_indices: torch.Tensor, page_size: int) -> np.ndarray:
    return (kv_indices[::page_size] // page_size).cpu().numpy()


def kv_to_page_num(num_kv_indices: int, page_size: int):
    return (num_kv_indices + page_size - 1) // page_size


def page_align_floor(length: int, page_size: int) -> int:
    return (length // page_size) * page_size


def free_swa_out_of_window_slots(
    req: Req,
    pre_len: int,
    *,
    sliding_window_size: int,
    page_size: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    is_chunk_cache: bool = False,
    retain_floor: int | None = None,
) -> None:
    if not req.kv.holds_kv:
        return

    # For swa radix cache, we need to evict the tokens that are not in the tree cache and also not in the sliding window
    assert req.kv.cache_protected_len % page_size == 0, (
        "cache_protected_len must be page aligned"
    )
    req.kv.swa_evicted_seqlen = max(
        req.kv.swa_evicted_seqlen, req.kv.swa_dead_lo(page_size)
    )

    if is_chunk_cache:
        # Chunk cache builds no radix tree, so no tombstone-leaf concern; evict
        # up to the window boundary (the trailing floor keeps it page-aligned).
        evict_threshold = pre_len - sliding_window_size
    else:
        # Radix cache: keep max(window, page). The trailing floor page-aligns the
        # frontier, and subtracting at least one page keeps it below the insert
        # boundary (page_floor(seq_len)) so the last leaf is never all-tombstone.
        # No extra page margin is needed.
        evict_threshold = pre_len - max(sliding_window_size, page_size)
    if retain_floor is not None and not is_chunk_cache:
        # The caller owns where the floor is (see BasePrefixCache.swa_retain_floor);
        # this only promises not to free past it. Chunk cache has no tree, so a
        # retained checkpoint could never be matched and holding it is pure cost.
        evict_threshold = min(evict_threshold, retain_floor)

    new_swa_evicted_seqlen = max(
        req.kv.swa_evicted_seqlen,
        evict_threshold,
    )

    if page_size > 1:
        new_swa_evicted_seqlen = (new_swa_evicted_seqlen // page_size) * page_size

    if new_swa_evicted_seqlen > req.kv.swa_evicted_seqlen:
        free_slots = req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, req.kv.swa_evicted_seqlen : new_swa_evicted_seqlen
        ]
        # Local import: the unified allocators import this module lazily for
        # eviction; a module-level import here would be a cycle hazard.
        from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
            UnifiedSWATokenToKVPoolAllocator,
        )

        if isinstance(token_to_kv_pool_allocator, UnifiedSWATokenToKVPoolAllocator):
            # Contiguous range with host-int bounds: hand the composite its
            # start position so the free stays host-sync-free (`free_segment`
            # derives page reps by stride math instead of `torch.unique`).
            token_to_kv_pool_allocator.free_swa(
                free_slots, start_pos=req.kv.swa_evicted_seqlen
            )
        else:
            token_to_kv_pool_allocator.free_swa(free_slots)
        req.kv.swa_evicted_seqlen = new_swa_evicted_seqlen


def free_kv_row_segments(
    allocator: BaseTokenToKVPoolAllocator,
    segments: list[tuple[torch.Tensor, int]],
    *,
    swa_evicted_seqlen: int,
) -> None:
    """Free ascending disjoint ``(kv_indices, start_pos)`` segments of one
    request's kv row, split at the SWA eviction floor."""
    swa_dead: list[tuple[torch.Tensor, int]] = []
    swa_alive: list[tuple[torch.Tensor, int]] = []
    for kv_indices, start_pos in segments:
        num_indices = kv_indices.numel()
        if num_indices == 0:
            continue
        # Below the floor the SWA peers are already gone -- window eviction, or
        # the deliberately unmapped prefix of a PD decode SWA-tail prealloc.
        num_dead = min(max(swa_evicted_seqlen - start_pos, 0), num_indices)
        if num_dead > 0:
            swa_dead.append((kv_indices[:num_dead], start_pos))
        if num_dead < num_indices:
            swa_alive.append((kv_indices[num_dead:], start_pos + num_dead))

    if swa_dead and swa_alive:
        # The two sides are separate calls, so neither one's page-disjointness
        # check sees a floor that splits a page between them.
        assert swa_evicted_seqlen % allocator.page_size == 0, (
            f"SWA eviction floor {swa_evicted_seqlen} splits a page "
            f"(page_size {allocator.page_size})"
        )
    if swa_dead:
        allocator.free_full_segments(swa_dead)
    if swa_alive:
        allocator.free_segments(swa_alive)


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict_for_alloc(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator: evict only the shortfall (mirrors the SWA arm)
        available_size = allocator.available_size()
        if available_size < num_tokens:
            tree_cache.evict_for_alloc(
                EvictParams(num_tokens=num_tokens - available_size)
            )


def retraction_backup(
    req: Req,
    tree_cache: BasePrefixCache,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    backend: str,
) -> bool:
    """Returns False when the host pool cannot hold the backup; the caller
    aborts the request since its KV cannot be preserved."""
    if backend == "cpu_tensor":
        req.offload_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)
        return True
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.seqlen <= 1:
        return True

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    req.kv.retraction_backup = unified_cache.retraction_backup(req)
    return req.kv.retraction_backup is not None


def retraction_restore(
    req: Req,
    tree_cache: BasePrefixCache,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    backend: str,
) -> None:
    if backend == "cpu_tensor":
        req.load_kv_cache(req_to_token_pool, token_to_kv_pool_allocator)
        return
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.seqlen <= 1:
        return

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    assert req.kv.retraction_backup is not None
    unified_cache.retraction_restore(req, req.kv.retraction_backup)
    req.kv.retraction_backup = None


def retraction_discard(req: Req, tree_cache: BasePrefixCache, backend: str) -> None:
    if backend == "cpu_tensor":
        req.kv.retraction_backup = None
        return
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.kv.retraction_backup is None:
        return

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    unified_cache.retraction_discard(req.kv.retraction_backup)
    req.kv.retraction_backup = None


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    assert (not req.kv.holds_kv) == req.kv.is_kv_released
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if not req.kv.holds_kv:
        assert tree_cache.supports_mamba(), (
            "Only MambaRadixCache allow freeing before alloc"
        )
        # TODO (csy, hanming): clean up this early allocation logic
        if req.kv.holds_mamba:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.kv.mamba_pool_idx.unsqueeze(-1)
            )
            req.kv.mamba_pool_idx = None
        return

    effective_kv_committed_len = req.effective_kv_committed_len()
    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
        kv_len_to_handle=effective_kv_committed_len,
    )

    # StreamingSession.cache_finished_req handles speculative tail trim
    # internally, then sets req_pool_idx = None.
    assert (not req.kv.holds_kv) == req.kv.is_kv_released
    if not req.kv.holds_kv:
        return

    start_p, end_p = effective_kv_committed_len, req.kv.kv_allocated_len
    _release_overallocated_kv_indices(req, start_p, end_p, tree_cache)

    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert req.kv.holds_mamba, (
            "mamba state is freed while the tree cache does not manage mamba states"
        )
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    tree_cache.req_to_token_pool.free(req)
    req.kv.mark_kv_released()


def _release_overallocated_kv_indices(
    req: Req, start_p: int, end_p: int, tree_cache: BasePrefixCache
) -> None:
    allocator = tree_cache.token_to_kv_pool_allocator
    page_size = allocator.page_size
    spec_algo = get_spec().speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not get_serving().strip_thinking_cache:
        assert start_p == end_p, (
            f"Unexpected overallocated KV cache, {req.kv.kv_committed_len=}, {req.kv.kv_allocated_len=}"
        )

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        # start_p is aligned to the allocator's physical page size above, so it
        # never shares a page with cache_finished_req's tail free in this group.
        tree_cache.free_kv_row(req.kv, [(start_p, end_p)])


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
