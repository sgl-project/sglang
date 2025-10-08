from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class ChunkCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size

    # NOTE (csy): this is to determine if a cache has prefix matching feature.
    # Chunk cache always return True to indicate no prefix matching.
    # TODO (csy): Using a prefix cache trait to replace this
    @property
    def disable(self):
        return True

    def reset(self):
        pass

    def match_prefix(self, **unused_kwargs) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
        )

    def cache_finished_req(self, req: Req, insert: bool = True):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            # For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids
            : len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool_allocator.free(kv_indices)

    def cache_unfinished_req(self, req: Req, chunked=False):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

    def evict(self, num_tokens: int):
        pass

    def inc_lock_ref(self, node: Any):
        return 0

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        return 0

    def pretty_print(self):
        return ""


class SWAChunkCache(ChunkCache):
    """ChunkCache with support for hybrid KV cache operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: SWATokenToKVPoolAllocator,
        page_size: int,
        sliding_window_size: int = -1,
    ):
        super().__init__(req_to_token_pool, token_to_kv_pool_allocator, page_size)
        assert isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        # Keep the most recent `sliding_window_size` tokens (if > 0) for SWA layers.
        self.sliding_window_size = int(sliding_window_size)

    def evict_swa(
        self,
        req: Req,
        prelen: int,
        attention_chunk_size: int,
    ):
        """Evict SWA KV entries that fall outside the sliding window.

        Original behavior (no sliding window) freed all complete chunks older
        than the current prelen. With a sliding window, we restrict eviction so
        that at least the most recent `sliding_window_size` tokens remain.

        Eviction alignment rules:
          - Only evict full chunks of size `attention_chunk_size`.
          - Never evict tokens within the last `sliding_window_size` tokens.
        """
        # Clamp to sliding window size as only need to keep max sliding_window_size tokens for SWA layer.
        if self.sliding_window_size != -1 and (
            attention_chunk_size is None
            or attention_chunk_size > self.sliding_window_size
        ):
            attention_chunk_size = self.sliding_window_size

        if prelen < req.evicted_seqlen_local + attention_chunk_size:
            return

        # Maximum chunk-aligned boundary we could evict up to under original policy.
        candidate_new = attention_chunk_size * (prelen // attention_chunk_size)

        if self.sliding_window_size > 0:
            # Oldest token index we must keep (exclusive eviction upper bound).
            # We want to retain tokens in [keep_start, prelen).
            keep_start = max(0, prelen - self.sliding_window_size)
            # Align eviction upper bound down to full chunks.
            keep_start_aligned = attention_chunk_size * (
                keep_start // attention_chunk_size
            )
            # We can only evict tokens strictly before keep_start_aligned.
            allowed_evict_upto = keep_start_aligned
            new_evicted_seqlen_local = min(candidate_new, allowed_evict_upto)
        else:
            new_evicted_seqlen_local = candidate_new

        # Ensure monotonic increase and chunk alignment.
        if new_evicted_seqlen_local <= req.evicted_seqlen_local:
            return

        free_slots = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, req.evicted_seqlen_local : new_evicted_seqlen_local
        ]
        self.token_to_kv_pool_allocator.free_swa(free_slots)
        req.evicted_seqlen_local = new_evicted_seqlen_local

    def evict(self, num_tokens: int):
        pass
