"""Radix cache for all-SWA models (every layer is sliding-window attention)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheFinishedReqResult,
    EvictParams,
    EvictResult,
    InsertParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

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
        assert (
            self.sliding_window_size is not None
        ), "sliding_window_size must be set for PureSWARadixCache"
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
        """No-op: PureSWARadixCache uses RadixCache's simple tree structure
        which doesn't need the dual-LRU sanity checks of SWARadixCache."""
        pass

    def evict(self, params: EvictParams) -> EvictResult:
        """For all-SWA models, evict_from_tree_cache passes swa_num_tokens
        (with num_tokens=0). Use whichever is non-zero."""
        num_tokens = max(params.num_tokens, params.swa_num_tokens)
        return super().evict(EvictParams(num_tokens=num_tokens))

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> CacheFinishedReqResult:
        """Cache request when it finishes.

        Only inserts the prefill portion [0, evict_floor) into the radix tree.
        The window portion [swa_evicted_seqlen, committed_len) is freed back
        to the allocator. The range [evict_floor, swa_evicted_seqlen) was already
        freed by _evict_swa during decode — we skip it to avoid double-free.
        """
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = kv_len_to_handle
        if self.disable:
            is_insert = False

        allocator_page = self.token_to_kv_pool_allocator.page_size
        assert self.page_size == allocator_page

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(allocator_page)
        keys_len = len(radix_key)

        old_prefix_len = req.cache_protected_len
        swa_evict_floor = req.swa_evict_floor
        swa_evicted_seqlen = req.kv.swa_evicted_seqlen

        if swa_evict_floor > 0:
            swa_evict_floor = (
                -(-swa_evict_floor // allocator_page) * allocator_page
            )

        if swa_evict_floor > 0 or swa_evicted_seqlen > swa_evict_floor:
            insert_end = min(swa_evict_floor, keys_len)
        else:
            insert_end = keys_len

        if is_insert and insert_end > 0:
            insert_values = kv_indices[:insert_end].to(dtype=torch.int64, copy=True)
            result = self.insert(
                InsertParams(key=radix_key[:insert_end], value=insert_values)
            )
            new_prefix_len = result.prefix_len
            if new_prefix_len > old_prefix_len:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[old_prefix_len:new_prefix_len]
                )
            alive_start = max(swa_evicted_seqlen, insert_end)
            if alive_start < keys_len:
                self.token_to_kv_pool_allocator.free(kv_indices[alive_start:keys_len])
        else:
            if swa_evicted_seqlen > swa_evict_floor:
                free_end = min(swa_evict_floor, keys_len)
                if old_prefix_len < free_end:
                    self.token_to_kv_pool_allocator.free(
                        kv_indices[old_prefix_len:free_end]
                    )
                alive_start = max(swa_evicted_seqlen, old_prefix_len)
                if alive_start < keys_len:
                    self.token_to_kv_pool_allocator.free(
                        kv_indices[alive_start:keys_len]
                    )
            elif old_prefix_len < keys_len:
                self.token_to_kv_pool_allocator.free(
                    kv_indices[old_prefix_len:keys_len]
                )

        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

        return CacheFinishedReqResult(unhandled_kv_start=keys_len)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """During chunked prefill, swa_evicted_seqlen is 0 and no SWA eviction
        has happened yet, so standard RadixCache logic is correct."""
        super().cache_unfinished_req(req, chunked=chunked)

    def available_and_evictable_str(self) -> str:
        allocator = self.token_to_kv_pool_allocator
        swa_available = allocator.swa_available_size()
        swa_evictable = self.swa_evictable_size()
        return (
            f"SWA available tokens: {swa_available + swa_evictable} "
            f"({swa_available=} + {swa_evictable=})\n"
        )
