from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


logger = logging.getLogger(__name__)


class ChunkCache(BasePrefixCache):
    """
    ChunkCache is used when radix cache is disabled.

    That includes standard chunked-prefill setups and the decode side of P/D
    disaggregation when decode radix cache is not enabled.
    """

    def __init__(self, params: CacheInitParams):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        self.protected_size_ = 0

    def is_chunk_cache(self) -> bool:
        return True

    # NOTE (csy): this is to determine if a cache has prefix matching feature.
    # Chunk cache always return True to indicate no prefix matching.
    # TODO (csy): Using a prefix cache trait to replace this
    @property
    def disable(self):
        return True

    def reset(self):
        pass

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        # ChunkCache does not support prefix caching, so insert is a no-op
        return InsertResult(prefix_len=0)

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ):
        # For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids
        # The protected prefix is not this req's to free.
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, req.cache_protected_len : kv_len_to_handle
        ]
        self.token_to_kv_pool_allocator.free(kv_indices)

    def cache_unfinished_req(self, req: Req, chunked=False):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ]
        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

    def evict(self, params: EvictParams) -> EvictResult:
        return EvictResult()

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        return IncLockRefResult(delta=0)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        return DecLockRefResult(delta=0)

    def protected_size(self):
        # NOTE: no protected size in chunk cache. Chunk cache's eviction is the same with request's lifecycle.
        return 0

    def pretty_print(self):
        return ""


class SWAChunkCache(ChunkCache):
    """ChunkCache with support for sliding window attention."""

    def __init__(self, params: CacheInitParams):
        # DeepSeek V4 HiSparse wraps SWATokenToKVPoolAllocator and exposes the same API.
        assert isinstance(
            params.token_to_kv_pool_allocator,
            (
                SWATokenToKVPoolAllocator,
                DeepSeekV4HiSparseTokenToKVPoolAllocator,
            ),
        )
        super().__init__(params)

        self.sliding_window_size = params.sliding_window_size
        self.chunked_prefill_size = params.chunked_prefill_size

    def supports_swa(self) -> bool:
        assert (
            self.sliding_window_size is not None
        ), "sliding_window_size must be set for SWAChunkCache"
        return True

    def evict(self, params: EvictParams) -> EvictResult:
        return EvictResult()


class PureSWAChunkCache(SWAChunkCache):
    """ChunkCache for all-SWA models (no full attention layers).

    For hybrid models, full_to_swa_index_mapping prevents SWA double-free.
    All-SWA models lack this mapping, so on request completion we must
    explicitly skip the range already freed by ``free_swa_out_of_window_slots``
    (a.k.a. _evict_swa) during decode.

    ``req.swa_evict_floor`` only protects the prompt/image KV while the request
    is active. ChunkCache does not retain finished prefixes, so the protected
    prefix is released here when the request finishes.
    """

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ):
        kv_committed_len = kv_len_to_handle
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]
        # As in the base class, the protected prefix is not this req's to free.
        protected_len = req.cache_protected_len
        evict_floor = req.swa_evict_floor
        evicted_seqlen = req.kv.swa_evicted_seqlen
        if evicted_seqlen > evict_floor:
            parts = []
            if evict_floor > protected_len:
                parts.append(kv_indices[protected_len:evict_floor])
            if evicted_seqlen < kv_committed_len:
                parts.append(
                    kv_indices[max(evicted_seqlen, protected_len) : kv_committed_len]
                )
            if parts:
                self.token_to_kv_pool_allocator.free(torch.cat(parts))
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[protected_len:])
