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


def _swa_evict_frontier(
    req: Req,
    pre_len: int,
    *,
    sliding_window_size: int,
    page_size: int,
    is_chunk_cache: bool,
    retain_floor: int | None = None,
) -> Optional[int]:
    """Host-only half of SWA eviction: advance the protected floor and return
    the new eviction frontier, or ``None`` when nothing can be freed yet.

    Split out so the batched variant below shares one copy of the boundary math.
    """
    # For swa radix cache, we need to evict the tokens that are not in the tree cache and also not in the sliding window
    assert (
        req.cache_protected_len % page_size == 0
    ), "cache_protected_len must be page aligned"
    evict_floor = max(req.cache_protected_len, getattr(req, "swa_evict_floor", 0))
    if page_size > 1 and evict_floor > req.cache_protected_len:
        evict_floor = -(-evict_floor // page_size) * page_size
    req.kv.swa_evicted_seqlen = max(req.kv.swa_evicted_seqlen, evict_floor)

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

    if new_swa_evicted_seqlen <= req.kv.swa_evicted_seqlen:
        return None
    return new_swa_evicted_seqlen


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
    if req.kv is None:
        return

    new_swa_evicted_seqlen = _swa_evict_frontier(
        req,
        pre_len,
        sliding_window_size=sliding_window_size,
        page_size=page_size,
        is_chunk_cache=is_chunk_cache,
        retain_floor=retain_floor,
    )
    if new_swa_evicted_seqlen is None:
        return

    free_slots = req_to_token_pool.req_to_token[
        req.req_pool_idx, req.kv.swa_evicted_seqlen : new_swa_evicted_seqlen
    ]
    # Both bounds are page-aligned and the whole range is still SWA-mapped, so
    # name the pages positionally instead of rediscovering them: free_swa()
    # masks the mapping, whose data-dependent shape forces a device sync.
    free_swa_segment = getattr(token_to_kv_pool_allocator, "free_swa_segment", None)
    if free_swa_segment is not None:
        free_swa_segment(free_slots, start_pos=req.kv.swa_evicted_seqlen)
    else:
        token_to_kv_pool_allocator.free_swa(free_slots)
    req.kv.swa_evicted_seqlen = new_swa_evicted_seqlen


def _gather_slot_ranges(
    req_to_token: torch.Tensor,
    ranges: list[tuple[int, int, int]],
    *,
    step: int = 1,
) -> torch.Tensor:
    """One gather for ``cat(req_to_token[row, start:end:step] for row, start, end)``.

    The index math runs on the device off one small host->device copy, so the
    number of gathered slots never turns into host work.

    ``step`` lets the caller gather one token per page instead of every token,
    which is what the sync-free release path needs (see
    ``free_swa_out_of_window_slots_batch``). At ``step=1`` the emitted ops are
    the same as before.
    """
    stride = req_to_token.shape[1]
    device = req_to_token.device
    lengths = [-(-(end - start) // step) for _, start, end in ranges]

    if len(ranges) == 1:
        # Single request (e.g. one-request prefill batch): a plain slice beats
        # building index tensors for it.
        row, start, end = ranges[0]
        return req_to_token[row, start:end:step]

    base = torch.tensor(
        [row * stride + start for row, start, _ in ranges],
        dtype=torch.int64,
        device=device,
    )
    if len(set(lengths)) == 1:
        # Common case: every request frees the same number of slots — one token
        # or one page per decode step, one chunk per prefill chunk.
        offsets = torch.arange(
            0, lengths[0] * step, step, dtype=torch.int64, device=device
        )
        flat = (base.unsqueeze(1) + offsets).view(-1)
    else:
        lengths_t = torch.tensor(lengths, dtype=torch.int64, device=device)
        range_start = torch.cumsum(lengths_t, 0) - lengths_t
        # base_i + (k - range_start_i) * step, folded so the per-element index
        # still costs one arange and one repeat_interleave.
        flat = torch.arange(
            int(sum(lengths)), dtype=torch.int64, device=device
        ) * step - torch.repeat_interleave(range_start * step - base, lengths_t)
    return req_to_token.view(-1)[flat]


def free_swa_out_of_window_slots_batch(
    reqs: list[Req],
    pre_lens: list[int],
    *,
    sliding_window_size: int,
    page_size: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    is_chunk_cache: bool = False,
    retain_floors: list[int | None] | None = None,
) -> None:
    """``free_swa_out_of_window_slots`` for a whole batch, in one release call.

    Two costs are being removed here, and they are independent.

    Calling per request puts O(batch_size) release calls on the scheduler thread
    right before it can launch the next forward — ~90us per request, i.e. ~185ms
    for a 2048-request decode batch. Gathering the whole batch's out-of-window
    range in one indexed read collapses that to one call.

    That one call still syncs if it is ``free_swa()``, which rediscovers the SWA
    pages by reading the mapping back (``unique`` + a ``> 0`` mask); both shapes
    are data-dependent, so the host blocks -- and behind the WAR-fenced schedule
    stream that means waiting out the whole in-flight forward, not the ~90us.
    Every range here is page-aligned at both ends (``_swa_evict_frontier`` floors
    the frontier and the previous frontier was floored the same way) and still
    fully SWA-mapped, so gather one token per page instead and let the allocator
    name the pages positionally: one op, no sync.
    """
    plan = []
    if retain_floors is None:
        retain_floors = [None] * len(reqs)
    for req, pre_len, retain_floor in zip(reqs, pre_lens, retain_floors):
        if req.kv is None:
            continue
        new_swa_evicted_seqlen = _swa_evict_frontier(
            req,
            pre_len,
            sliding_window_size=sliding_window_size,
            page_size=page_size,
            is_chunk_cache=is_chunk_cache,
            retain_floor=retain_floor,
        )
        if new_swa_evicted_seqlen is None:
            continue
        plan.append((req, req.kv.swa_evicted_seqlen, new_swa_evicted_seqlen))

    if not plan:
        return

    req_to_token = req_to_token_pool.req_to_token
    ranges = [(req.req_pool_idx, start, end) for req, start, end in plan]
    free_swa_page_reps = getattr(token_to_kv_pool_allocator, "free_swa_page_reps", None)
    if free_swa_page_reps is not None and page_size > 1:
        free_swa_page_reps(_gather_slot_ranges(req_to_token, ranges, step=page_size))
    else:
        token_to_kv_pool_allocator.free_swa(_gather_slot_ranges(req_to_token, ranges))
    for req, _, new_swa_evicted_seqlen in plan:
        req.kv.swa_evicted_seqlen = new_swa_evicted_seqlen


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
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator: evict only the shortfall (mirrors the SWA arm)
        available_size = allocator.available_size()
        if available_size < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens - available_size))


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
    req.retraction_backup = unified_cache.retraction_backup(req)
    return req.retraction_backup is not None


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
    assert req.retraction_backup is not None
    unified_cache.retraction_restore(req, req.retraction_backup)
    req.retraction_backup = None


def retraction_discard(req: Req, tree_cache: BasePrefixCache, backend: str) -> None:
    if backend == "cpu_tensor":
        req.retraction_backup = None
        return
    if backend != "host_pool":
        raise ValueError(f"Unknown retraction backup backend: {backend}")
    if req.retraction_backup is None:
        return

    unified_cache = cast("UnifiedRadixCache", tree_cache)
    unified_cache.retraction_discard(req.retraction_backup)
    req.retraction_backup = None


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # the two resources currently have the same lifecycle, thus simplify logic below
    assert (req.req_pool_idx is None) == (req.kv is None)
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    effective_kv_committed_len = req.effective_kv_committed_len()
    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
        kv_len_to_handle=effective_kv_committed_len,
    )

    # StreamingSession.cache_finished_req handles speculative tail trim
    # internally, then sets req_pool_idx = None.
    assert (req.req_pool_idx is None) == (req.kv is None)
    if req.req_pool_idx is None and req.kv is None:
        return

    start_p, end_p = effective_kv_committed_len, req.kv.kv_allocated_len
    _release_overallocated_kv_indices(req, start_p, end_p, tree_cache)

    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    tree_cache.req_to_token_pool.free(req)
    req.kv = None


def _release_overallocated_kv_indices(
    req: Req, start_p: int, end_p: int, tree_cache: BasePrefixCache
) -> None:
    allocator = tree_cache.token_to_kv_pool_allocator
    page_size = allocator.page_size
    spec_algo = get_spec().speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not get_serving().strip_thinking_cache:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        # start_p is aligned to the allocator's physical page size above, so it
        # never shares a page with cache_finished_req's tail free in this group.
        free_swa_segment = getattr(allocator, "free_swa_segment", None)
        if (
            free_swa_segment is not None
            and page_size > 1
            and tree_cache.is_chunk_cache()
        ):
            # This tail sits at the end of the sequence, i.e. inside the SWA
            # window, so both pools still hold it and can be named positionally
            # (free_segment would fall back to the syncing discovery path).
            # Gated on page_size > 1: that is where naming the pages pays, and it
            # also excludes PureSWATokenToKVPoolAllocator (always page_size == 1),
            # whose full_attn_allocator IS its swa_attn_allocator -- splitting the
            # release there would push the same slots onto the free list twice.
            allocator.free_full_segment(indices_to_free, start_pos=start_p)
            free_swa_segment(indices_to_free, start_pos=start_p)
        else:
            allocator.free_segment(indices_to_free, start_pos=start_p)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    return tree_cache.available_and_evictable_str()
