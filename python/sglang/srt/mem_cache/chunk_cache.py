from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class ChunkCacheEntry:
    def __init__(self, rid: str, value: torch.Tensor):
        self.rid = rid
        self.value = value


class ChunkCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        page_size: int,
        token_to_kv_pool_allocator_local: Optional[TokenToKVPoolAllocator] = None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.token_to_kv_pool_allocator_local = token_to_kv_pool_allocator_local

    def reset(self):
        pass

    def match_prefix(self, **unused_kwargs) -> Tuple[List[int], int]:
        return [], None

    def cache_finished_req(self, req: Req):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            # For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids
            : len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool_allocator.free(kv_indices)
        if self.token_to_kv_pool_allocator_local is not None:
            kv_indices_local = self.req_to_token_pool.req_to_token_local[
                req.req_pool_idx,
                req.evicted_seqlen_local : len(req.origin_input_ids)
                + len(req.output_ids)
                - 1,
            ]
            self.token_to_kv_pool_allocator_local.free(kv_indices_local)

    def cache_unfinished_req(self, req: Req):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = kv_indices
        if self.token_to_kv_pool_allocator_local is not None:
            kv_indices_local = self.req_to_token_pool.req_to_token_local[
                req.req_pool_idx, : len(req.fill_ids)
            ]
            req.prefix_indices_local = kv_indices_local

    def insert(self):
        raise NotImplementedError()

    def evict_hybrid(
        self,
        req: Req,
        prelen: int,
        attention_chunk_size: int,
    ):
        if prelen >= req.evicted_seqlen_local + attention_chunk_size:
            new_evicted_seqlen_local = attention_chunk_size * (
                prelen // attention_chunk_size
            )
            free_slots = self.req_to_token_pool.req_to_token_local[
                req.req_pool_idx, req.evicted_seqlen_local : new_evicted_seqlen_local
            ]
            self.token_to_kv_pool_allocator_local.free(free_slots)
            req.evicted_seqlen_local = new_evicted_seqlen_local

    def evict(self, num_tokens: int):
        pass

    def inc_lock_ref(self, node: Any):
        return 0

    def dec_lock_ref(self, node: Any):
        return 0

    def pretty_print(self):
        return ""
