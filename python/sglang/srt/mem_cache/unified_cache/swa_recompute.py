"""Cache side of DeepSeek-V4 SWA recompute: bind fresh SWA rows to a FULL
prefix that outlived its window, for the worker's replay to fill."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import UnifiedSWAAllocatorBase
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

# A replay costs about a prefill of swa_recompute_len tokens; prefixes that
# save less than twice that are prefilled again instead.
_MIN_GAIN_RATIO = 2


def swa_recompute_len(*, sliding_window: int, num_layers: int, page_size: int) -> int:
    """Tokens to replay so every layer's last window is exact: layer ``l`` reads
    window rows built from layer ``l - 1`` outputs, one window further back each."""
    return ceil_align(
        ceil_align(sliding_window, page_size) + (num_layers - 1) * sliding_window,
        page_size,
    )


class SWARecompute(msgspec.Struct):
    # First prefix position the worker replays; page-aligned.
    start: int
    # FULL ids from ``start`` whose SWA rows only the replay reads; freed after the forward.
    workspace: torch.Tensor
    # False once a batch took the replay.
    pending: bool = True

    def take_start(self) -> Optional[int]:
        pending, self.pending = self.pending, False
        return self.start if pending else None


def supports_swa_recompute(allocator) -> bool:
    return (
        isinstance(allocator, SWATokenToKVPoolAllocator)
        and not isinstance(allocator, UnifiedSWAAllocatorBase)
        and not allocator.swa_req_ring
    )


def _recomputable_full_len(
    cache: UnifiedRadixCache, key: RadixKey, matched_len: int
) -> tuple[int, Optional[NodeId]]:
    """Device FULL length a replay can reach from ``matched_len``, else ``matched_len``."""
    replay_len = cache.swa_recompute_len
    window = ceil_align(cache.sliding_window_size, cache.page_size)
    full_len, node_id, _ = cache.tree_core.match_full_device_prefix(key)
    full_len = full_len // cache.page_size * cache.page_size
    if full_len - matched_len < _MIN_GAIN_RATIO * replay_len:
        return matched_len, None
    window_start = full_len - window
    if cache.tree_core.swa_tombstone_ranges(key, window_start, full_len) != [
        (window_start, full_len)
    ]:
        return matched_len, None
    return full_len, node_id


def swa_recompute_hit_length(
    cache: UnifiedRadixCache, key: RadixKey, matched_len: int, full_kv_hit_length: int
) -> int:
    # full_kv_hit_length also counts host FULL, so it only bounds the gain.
    if full_kv_hit_length - matched_len < _MIN_GAIN_RATIO * cache.swa_recompute_len:
        return 0
    full_len, _ = _recomputable_full_len(cache, key, matched_len)
    return full_len - matched_len


def init_swa_recompute(
    cache: UnifiedRadixCache, req: Req, key: RadixKey
) -> Optional[tuple[int, NodeId, SWARecompute]]:
    """Bind fresh SWA rows behind the FULL prefix and publish its last window.
    The window holds no data until the returned replay runs, so the request must
    enter the batch being built."""
    matched_len = req.prefix_len
    full_len, node_id = _recomputable_full_len(cache, key, matched_len)
    if node_id is None:
        return None
    replay_len = cache.swa_recompute_len
    window = ceil_align(cache.sliding_window_size, cache.page_size)
    allocator = cache.token_to_kv_pool_allocator
    tree_core = cache.tree_core

    tree_core.inc_full_pin(node_id)
    try:
        shortfall = replay_len - allocator.swa_available_size()
        if shortfall > 0:
            cache.evict_for_alloc(EvictParams(swa_num_tokens=shortfall))
        if allocator.swa_available_size() < replay_len:
            return None
        gained = tree_core.collect_full_device_indices(node_id, req.last_node)[
            : full_len - matched_len
        ]
        replayed = gained[-replay_len:]
        swa_indices = allocator.swa_attn_allocator.alloc(replay_len)
        assert swa_indices is not None
        allocator.set_full_to_swa_mapping(replayed, swa_indices)
        for action in tree_core.attach_swa_window(
            key,
            full_len - window,
            full_len,
            swa_indices[-window:].to(torch.int64),
        ):
            cache._apply_cache_action(action)
    finally:
        tree_core.dec_full_pin(node_id)

    match = cache.match_prefix(MatchPrefixParams(key=key))
    assert match.device_prefix_len == full_len, (
        f"SWA recompute published [{full_len - window}, {full_len}) but the "
        f"prefix matches {match.device_prefix_len} tokens"
    )
    return (
        full_len - matched_len,
        match.last_device_node,
        SWARecompute(start=full_len - replay_len, workspace=replayed[:-window]),
    )


def release_swa_recompute_workspace(cache: UnifiedRadixCache, req: Req) -> None:
    recompute = req.swa_recompute
    if recompute is None:
        return
    req.swa_recompute = None
    # Whole pages from a page-aligned start: the segment free needs no torch.unique.
    cache.token_to_kv_pool_allocator.free_swa_segment(
        recompute.workspace, start_pos=recompute.start
    )
