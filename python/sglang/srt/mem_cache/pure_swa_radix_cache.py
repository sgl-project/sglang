"""Radix cache for all-SWA models (every layer is sliding-window attention)."""

from __future__ import annotations

import logging

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache

logger = logging.getLogger(__name__)


class PureSWARadixCache(RadixCache):
    """Radix cache for all-SWA models (no full attention layers).

    Extends RadixCache with SWA semantics. Only caches the prefill portion
    [0, evict_floor) on request completion. Window-range KV is freed.
    No tombstone mechanism needed.
    """

    def __init__(self, params: CacheInitParams):
        super().__init__(params)
        self.sliding_window_size = params.sliding_window_size

    def supports_swa(self) -> bool:
        assert self.sliding_window_size is not None, (
            "sliding_window_size must be set for PureSWARadixCache"
        )
        return True

    def swa_evictable_size(self):
        return self.evictable_size_

    def swa_protected_size(self):
        return self.protected_size_

    def full_evictable_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def sanity_check(self):
        """No-op: an all-SWA model has no full tier, so there is no full/SWA
        split to cross-check."""
        pass

    def evict(self, params: EvictParams) -> EvictResult:
        """For all-SWA models, evict_from_tree_cache passes swa_num_tokens
        (with num_tokens=0). Use whichever is non-zero."""
        num_tokens = max(params.num_tokens, params.swa_num_tokens)
        return super().evict(EvictParams(num_tokens=num_tokens))

    def available_and_evictable_str(self) -> str:
        allocator = self.token_to_kv_pool_allocator
        swa_available = allocator.swa_available_size()
        swa_evictable = self.swa_evictable_size()
        return (
            f"SWA available tokens: {swa_available + swa_evictable} "
            f"({swa_available=} + {swa_evictable=})\n"
        )
