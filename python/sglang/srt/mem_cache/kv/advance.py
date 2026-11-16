"""Effects of a request advancing: past a state checkpoint boundary, and
past the rows a window no longer reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import msgspec

from sglang.srt.mem_cache.kv.alloc.demand import retained_window_tokens

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class TrackingAdvance(msgspec.Struct, frozen=True, kw_only=True):
    """The completed forward's checkpoint and the next forward's destination.

    An omitted boundary preserves the latest checkpoint; None invalidates it
    before an in-place write. Boundaries come from the completed batch snapshot.
    """

    boundary: Union[int, None, msgspec.UnsetType] = msgspec.UNSET
    last_position: Optional[int] = None
    next_position: Optional[int] = None
    preserve_previous: bool = False
    release_positions: Tuple[int, ...] = ()


def advance_tracking(storage, req, update: TrackingAdvance) -> None:
    kv = req.kv
    for position in update.release_positions:
        if not kv.mamba_ping_pong_empty[position]:
            storage.release_tracking_slot(req, position)
    if update.preserve_previous:
        kv.mamba_prev_track_seqlen = kv.mamba_last_track_seqlen
    if update.boundary is not msgspec.UNSET:
        kv.mamba_last_track_seqlen = update.boundary
    if update.last_position is not None:
        kv.mamba_last_track_idx = update.last_position
    if update.next_position is not None:
        kv.mamba_next_track_idx = update.next_position


def coalesce_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge adjacent half-open ranges so a split that falls mid-page frees that page once."""
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def retire_window_rows(
    req: Req,
    pre_len: int,
    *,
    window_sub_pool: str,
    sliding_window_size: int,
    page_size: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    supports_prefix_sharing: bool = True,
    retain_floor: int | None = None,
    free_segment: Callable[..., None] | None = None,
    eviction_interval: int = 1,
) -> None:
    """Free the rows of a request's ``window_sub_pool`` that fell out of its
    window since its cursor, keeping what a prefix-sharing cache needs and
    nothing past ``retain_floor``; the cursor moves to the new boundary."""
    if not req.kv.holds_kv:
        return

    assert req.kv.cache_protected_len % page_size == 0, (
        "cache_protected_len must be page aligned"
    )
    # Protected rows limit what can be freed, not where the interval starts.
    evicted_seqlen = req.kv.get_evicted_seqlen(window_sub_pool)
    if pre_len - sliding_window_size < evicted_seqlen + eviction_interval:
        return
    from sglang.srt.layers.cp.base import is_cp_sharded_padded

    cp_sharded = is_cp_sharded_padded()
    window_page_size = page_size
    if cp_sharded:
        from sglang.srt.layers.cp.utils import from_cp_sharded

        window_page_size = from_cp_sharded(window_page_size)
    evicted_seqlen = max(evicted_seqlen, req.kv.window_dead_lo(window_page_size))
    req.kv.set_evicted_seqlen(window_sub_pool, evicted_seqlen)

    # Without prefix sharing nothing enters a tree: evict up to the window
    # (the trailing floor keeps it page-aligned). A prefix-sharing cache keeps
    # whole pages, so the last leaf, below page_floor(seq_len), is never
    # all-tombstone.
    evict_threshold = pre_len - retained_window_tokens(
        sliding_window_size,
        page_size,
        supports_prefix_sharing=supports_prefix_sharing,
    )
    if retain_floor is not None and supports_prefix_sharing:
        # The caller owns where the floor is (see BasePrefixCache.swa_retain_floor);
        # this only promises not to free past it. Without prefix sharing a retained
        # checkpoint could never be matched, so holding it is pure cost.
        evict_threshold = min(evict_threshold, retain_floor)

    new_evicted_seqlen = max(evicted_seqlen, evict_threshold)

    if page_size > 1:
        new_evicted_seqlen = (new_evicted_seqlen // page_size) * page_size
    if cp_sharded:
        # Release complete logical groups so every rank frees the same span.
        group_size = from_cp_sharded(page_size)
        new_evicted_seqlen = (new_evicted_seqlen // group_size) * group_size

    if new_evicted_seqlen > evicted_seqlen:
        free_slots = req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, evicted_seqlen:new_evicted_seqlen
        ]
        if free_segment is None:
            free_segment = token_to_kv_pool_allocator.free_swa_segment
        if cp_sharded:
            from sglang.srt.layers.cp.utils import (
                get_cp_sharded_padded,
                get_cp_sharded_padded_cache_indices,
            )

            free_slots = get_cp_sharded_padded_cache_indices(free_slots)
            evicted_seqlen = get_cp_sharded_padded(evicted_seqlen)
        free_segment(free_slots, start_pos=evicted_seqlen)
        req.kv.set_evicted_seqlen(window_sub_pool, new_evicted_seqlen)
