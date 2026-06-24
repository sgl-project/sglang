"""Pure SWA Radix Cache for all-SWA models (no full attention layers).

For models like UNLIMITED-OCR where every layer uses sliding window attention,
this cache extends RadixCache with SWA-specific semantics:
- Reports swa_evictable/protected sizes for memory accounting
- Only caches the prefill portion [0, evict_floor) on request completion
- Window-range KV is freed back to allocator (not cached)
- No tombstone mechanism needed (unlike SWARadixCache)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class PureSWARadixCache(RadixCache):
    """Radix cache for all-SWA models (no full attention layers).

    Extends RadixCache with SWA semantics. Only caches the prefill portion
    [0, evict_floor) on request completion. Window-range KV is freed.
    No tombstone mechanism needed.
    """

    def __init__(self, params: CacheInitParams):
        """Initialize PureSWARadixCache."""
        super().__init__(params)
        self.sliding_window_size = params.sliding_window_size

    def supports_swa(self) -> bool:
        """Return True if this cache supports SWA."""
        assert (
            self.sliding_window_size is not None
        ), "sliding_window_size must be set for PureSWARadixCache"
        return True

    # For all-SWA: all evictable/protected tokens are SWA tokens
    def swa_evictable_size(self):
        """Return evictable size for SWA tokens."""
        return self.evictable_size_

    def swa_protected_size(self):
        """Return protected size for SWA tokens."""
        return self.protected_size_

    def full_evictable_size(self):
        """Return 0 since all-SWA models have no full attention tokens."""
        return 0

    def full_protected_size(self):
        """Return 0 since all-SWA models have no full attention tokens."""
        return 0

    def sanity_check(self):
        """No-op: PureSWARadixCache uses RadixCache's simple tree structure
        which doesn't need the dual-LRU sanity checks of SWARadixCache."""
        pass

    def evict(self, params: EvictParams) -> EvictResult:
        """Evict tokens from the cache.

        For all-SWA models, evict_from_tree_cache passes swa_num_tokens
        (with num_tokens=0). We use whichever is non-zero.
        """
        # evict_from_tree_cache sets num_tokens=0 and swa_num_tokens=N for all-SWA
        num_tokens = max(params.num_tokens, params.swa_num_tokens)
        adjusted_params = EvictParams(num_tokens=num_tokens)
        return super().evict(adjusted_params)

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes.

        Only inserts the prefill portion [0, evict_floor) into the radix tree.
        The window portion [swa_evicted_seqlen, committed_len) is freed back
        to the allocator since it's unlikely to be reused.
        The range [evict_floor, swa_evicted_seqlen) was already freed by
        _evict_swa during decode — we skip it to avoid double-free.
        """
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            # _evict_swa zeros freed entries in req_to_token,
            # free_swa filters index > 0 → no double-free
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        # Build the (optionally bigram) page-aligned radix key. `key_len` plays
        # the role of `len(keys)` in the original wheel implementation.
        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        old_prefix_len = req.cache_protected_len

        # Compute the boundary of what was freed by _evict_swa
        swa_evict_floor = req.swa_evict_floor
        swa_evicted_seqlen = req.swa_evicted_seqlen

        # Page-align evict_floor upward
        if self.page_size > 1 and swa_evict_floor > 0:
            swa_evict_floor = (
                -(-swa_evict_floor // self.page_size) * self.page_size
            )

        # Memory layout at request completion:
        #   [0, old_prefix_len)                    — already in tree (protected)
        #   [old_prefix_len, swa_evict_floor)      — alive, not in tree, not freed
        #   [swa_evict_floor, swa_evicted_seqlen)  — already freed by _evict_swa
        #   [swa_evicted_seqlen, key_len)          — alive (window), not freed
        #   [key_len, kv_committed_len)            — unaligned tail
        #
        # We insert [0, insert_end) into the tree, where insert_end = swa_evict_floor
        # (the cacheable prefill prefix). Everything else alive gets freed.

        if swa_evict_floor > 0:
            insert_end = min(swa_evict_floor, key_len)
        else:
            # No SWA eviction happened (e.g., very short request) — cache everything
            insert_end = key_len

        if is_insert and insert_end > 0:
            insert_values = kv_indices[:insert_end].to(dtype=torch.int64, copy=True)
            result = self.insert(
                InsertParams(key=radix_key[:insert_end], value=insert_values)
            )
            new_prefix_len = result.prefix_len
            # Free duplicates that were already in the tree
            dup_count = max(0, new_prefix_len - old_prefix_len)
            if dup_count > 0:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[old_prefix_len:new_prefix_len]
                )
            # Free alive-but-not-cached window tokens
            # [swa_evicted_seqlen, key_len) — window range, still alive, not cached
            alive_start = max(swa_evicted_seqlen, insert_end)
            if alive_start < key_len:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[alive_start:key_len]
                )
        else:
            # No insert: free all alive tokens that weren't freed by _evict_swa
            # [old_prefix_len, swa_evict_floor) — alive, never freed
            free_end = min(swa_evict_floor, key_len) if swa_evict_floor > 0 else key_len
            if free_end > old_prefix_len:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[old_prefix_len:free_end]
                )
            # [swa_evicted_seqlen, key_len) — alive window tokens
            alive_start = max(swa_evicted_seqlen, old_prefix_len)
            if swa_evicted_seqlen > 0 and alive_start < key_len:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[alive_start:key_len]
                )

        # Free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[key_len:])

        # Release the cache lock
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished (chunked prefill).

        During chunked prefill, swa_evicted_seqlen is 0 and no SWA eviction
        has happened yet, so standard RadixCache logic is correct.
        """
        super().cache_unfinished_req(req, chunked=chunked)

    def available_and_evictable_str(self) -> str:
        """Return human-readable string of available and evictable SWA tokens."""
        allocator = self.token_to_kv_pool_allocator
        swa_available = allocator.swa_available_size()
        swa_evictable = self.swa_evictable_size()
        return (
            f"SWA available tokens: {swa_available + swa_evictable} "
            f"({swa_available=} + {swa_evictable=})\n"
        )
