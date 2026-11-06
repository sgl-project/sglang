from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    RadixCacheWalkResult,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


def walk_radix_cache_for_canary(
    *,
    radix_cache: BasePrefixCache,
    unlocked_only: bool = False,
    swa_resident_only: bool = False,
) -> RadixCacheWalkResult:
    """Walk the radix tree and emit flat (slot_indices, positions, prev_slot_indices) tensors.

    With both flags False (default), emits every slot held by the radix cache (including slots
    also referenced by a currently-running req — that overlap is harmless redundancy with the
    per-forward HEAD/TAIL path). ``unlocked_only=True`` skips nodes still locked by a running
    req. ``swa_resident_only=True`` skips SWA-tombstoned nodes (slots evicted from the SWA
    window)."""
    if type(radix_cache) is not UnifiedRadixCache:
        raise NotImplementedError(
            f"walk_radix_cache_for_canary does not support {type(radix_cache).__name__}"
        )
    return radix_cache.tree_core.walk_for_kv_canary(
        unlocked_only=unlocked_only, swa_resident_only=swa_resident_only
    )
