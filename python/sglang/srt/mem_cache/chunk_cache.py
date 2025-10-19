from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)

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
        # KVPress: When compression is enabled, decode slots are written at logical positions
        # (seq_lens), not compressed positions (actual_kv_len). So we need to read the entire
        # req_to_token array up to the logical length.
        if req.actual_kv_len is not None:
            # Use logical length to capture decode slots at positions [seq_len, seq_len+1, ...]
            # Note: Same as non-KVPress case, the last output token's KV is not stored yet
            kv_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
            logger.info(
                f"[KVPress Debug] cache_finished_req for {req.rid}: "
                f"actual_kv_len={req.actual_kv_len}, origin_input_len={len(req.origin_input_ids)}, "
                f"output_ids_len={len(req.output_ids)}, logical_kv_len={kv_len}"
            )
        else:
            # For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids
            kv_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len
        ]
        logger.info(
            f"[KVPress Debug] cache_finished_req: "
            f"kv_indices_before_filter={kv_indices.tolist()}"
        )
        # Filter out zeros (compressed/pruned slots or padding)
        kv_indices = kv_indices[kv_indices != 0]
        logger.info(
            f"[KVPress Debug] cache_finished_req freeing {len(kv_indices)} slots: {kv_indices.tolist()}"
        )
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
