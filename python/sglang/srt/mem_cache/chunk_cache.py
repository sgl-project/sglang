from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


class ChunkCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        self.protected_size_ = 0

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

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        kv_committed_len = req.pop_committed_kv_cache()
        # For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
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

    def protected_size(self):
        # NOTE: no protected size in chunk cache. Chunk cache's eviction is the same with request's lifecycle.
        return 0

    def pretty_print(self):
        return ""


class SWAChunkCache(ChunkCache):
    """ChunkCache with support for hybrid KV cache operations."""

    def __init__(self, params: CacheInitParams):
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        super().__init__(params)
        self.is_local_attention = params.is_local_attention

    def evict_swa(
        self,
        req: Req,
        prelen: int,
        attention_chunk_size: int,
    ):
        thresh = req.evicted_seqlen_local + attention_chunk_size * 2
        if self.is_local_attention:
            thresh -= attention_chunk_size

        if prelen >= thresh:
            new_evicted_seqlen_local = (
                prelen // attention_chunk_size * attention_chunk_size
            ) - (attention_chunk_size if not self.is_local_attention else 0)
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.evicted_seqlen_local : new_evicted_seqlen_local
            ]
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            req.evicted_seqlen_local = new_evicted_seqlen_local

    def evict(self, num_tokens: int):
        pass
