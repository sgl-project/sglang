from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.allocator import (
    HIERARCHICAL_NSA_DECODE_MAX_TOKENS,
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
    is_enable_hierarchical_nsa,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


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

        if is_enable_hierarchical_nsa(self.token_to_kv_pool_allocator):
            if len(req.origin_input_ids) >= HIERARCHICAL_NSA_DECODE_MAX_TOKENS:
                kv_free_len = ceil_align(
                    HIERARCHICAL_NSA_DECODE_MAX_TOKENS, self.page_size
                )
            else:
                kv_free_len = kv_committed_len

            index_k_free_len = len(req.origin_input_ids) + max(
                len(req.output_ids) - 1, 0
            )

            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_free_len
            ]
            index_k_indices = self.req_to_token_pool.req_to_nsa_index_k[
                req.req_pool_idx, :index_k_free_len
            ]
            self.req_to_token_pool.free(req.req_pool_idx)
            self.protected_size_ -= len(req.prefix_indices)
            self.token_to_kv_pool_allocator.free((kv_indices, index_k_indices))

            logger.info(
                f"Free KV and index_k cache for request {req.rid}: "
                f"kv_shape={kv_indices.shape}, index_k_shape={index_k_indices.shape}, "
                f"kv_committed={kv_committed_len}, prompt_len={len(req.origin_input_ids)}, "
                f"output_len={len(req.output_ids)}"
            )
        else:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]

            self.req_to_token_pool.free(req.req_pool_idx)
            self.protected_size_ -= len(req.prefix_indices)
            self.token_to_kv_pool_allocator.free(kv_indices)

            logger.info(
                f"Free KV cache for request {req.rid}: kv_shape={kv_indices.shape}"
            )

    def cache_unfinished_req(self, req: Req, chunked=False):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        self.protected_size_ += len(kv_indices) - len(req.prefix_indices)

        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

        # For NSA: also save Index K indices for chunked prefill
        if is_enable_hierarchical_nsa(self.token_to_kv_pool_allocator):
            index_k_indices = self.req_to_token_pool.req_to_nsa_index_k[
                req.req_pool_idx, : len(req.fill_ids)
            ]
            req.index_k_prefix_indices = index_k_indices.to(
                dtype=torch.int32, copy=True
            )

    def evict(self, num_tokens: int):
        pass

    def inc_lock_ref(self, node: Any):
        return 0

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[str] = None):
        return 0

    def protected_size(self):
        return self.protected_size_

    def pretty_print(self):
        return ""


class SWAChunkCache(ChunkCache):
    """ChunkCache with support for hybrid KV cache operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: SWATokenToKVPoolAllocator,
        page_size: int,
    ):
        super().__init__(req_to_token_pool, token_to_kv_pool_allocator, page_size)
        assert isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)

    def evict_swa(
        self,
        req: Req,
        prelen: int,
        attention_chunk_size: int,
    ):
        if prelen >= req.evicted_seqlen_local + attention_chunk_size:
            new_evicted_seqlen_local = attention_chunk_size * (
                prelen // attention_chunk_size
            )
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.evicted_seqlen_local : new_evicted_seqlen_local
            ]
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            req.evicted_seqlen_local = new_evicted_seqlen_local

    def evict(self, num_tokens: int):
        pass
