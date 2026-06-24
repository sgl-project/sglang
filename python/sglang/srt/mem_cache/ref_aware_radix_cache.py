from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.ref_aware_cache_mixin import (
    TIER_HIGH_REF,
    TIER_LOW_REF,
    TIER_UNUSED,
    RefAwareCacheMixin,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


class RefAwareRadixCache(RefAwareCacheMixin, RadixCache):
    """RadixCache with priority-aware tiered eviction (no host pool)."""

    def __init__(self, params: CacheInitParams, server_args: ServerArgs = None):
        self._init_ref_aware_state(server_args)
        super().__init__(params)

    def reset(self):
        self._reset_ref_aware_state()
        super().reset()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        if self._evict_scope_stack:
            allow_low, allow_high = self._evict_scope_stack[-1]
        else:
            allow_low = True
            allow_high = False

        num_evicted = self._evict_tiered(params.num_tokens, allow_low, allow_high)
        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_tiered(self, num_tokens: int, allow_low: bool, allow_high: bool) -> int:
        num_evicted = 0
        num_evicted += self._evict_from_tier_heap(
            num_tokens - num_evicted,
            self.unused_evictable_leaves,
            TIER_UNUSED,
            self._evict_one_device,
        )
        if allow_low and num_evicted < num_tokens:
            num_evicted += self._evict_from_tier_heap(
                num_tokens - num_evicted,
                self.low_ref_evictable_leaves,
                TIER_LOW_REF,
                self._evict_one_device,
            )
        if allow_high and num_evicted < num_tokens:
            num_evicted += self._evict_from_tier_heap(
                num_tokens - num_evicted,
                self.high_ref_evictable_leaves,
                TIER_HIGH_REF,
                self._evict_one_device,
            )
        return num_evicted

    def _evict_one_device(self, node) -> int:
        self.token_to_kv_pool_allocator.free(node.value)
        num = len(node.value)
        self._delete_leaf(node)
        self._record_remove_event(node)
        return num
