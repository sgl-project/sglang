from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

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
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


logger = logging.getLogger(__name__)


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
        )

    def insert(self, params: InsertParams) -> InsertResult:
        # ChunkCache does not support prefix caching, so insert is a no-op
        return InsertResult(prefix_len=0)

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        kv_committed_len = req.pop_committed_kv_cache()
        # For decode server: if req.output_ids is empty, we want to free all req.origin_input_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]
        self.token_to_kv_pool_allocator.free(kv_indices)

    def cache_unfinished_req(self, req: Req, chunked=False):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
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
        assert isinstance(params.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
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


class SWAC4C128ChunkCache(ChunkCache):
    """ChunkCache with support for hybrid KV cache operations."""

    def __init__(self, params: CacheInitParams):
        from sglang.srt.hardware_backend.npu.hybrid_swa_c4_c128_memory_pool import (
            SWAC4C128TokenToKVPoolAllocator,
        )

        assert isinstance(
            params.token_to_kv_pool_allocator, SWAC4C128TokenToKVPoolAllocator
        )
        super().__init__(params)

        assert (
            params.sliding_window_size is not None
        ), "Sliding window size must be set for SWAC4C128ChunkCache"

        self.sliding_window_size = params.sliding_window_size
        self.compress_state_window_sizes = {
            "c4": 8 + 16,  # + draft_num_tokens
            "c128": 128 + 64,
            "swa": 128 + 64,
        }

    def evict_swa_c4c128_state(
        self,
        req: Req,
        prelen: int,
    ):
        new_evicted_seqlen_swa = max(
            req.swa_alloc_offset,
            (req.kv_committed_len - self.compress_state_window_sizes["swa"])
            // self.page_size
            * self.page_size,
        )
        new_evicted_seqlen_c4_state = max(
            req.c4_alloc_offset,
            (req.kv_committed_len - self.compress_state_window_sizes["c4"])
            // self.page_size
            * self.page_size,
        )
        new_evicted_seqlen_c128_state = max(
            req.c128_alloc_offset,
            (req.kv_committed_len - self.compress_state_window_sizes["c128"])
            // self.page_size
            * self.page_size,
        )

        if new_evicted_seqlen_swa > req.swa_alloc_offset:
            free_slots_swa = self.req_to_token_pool.req_to_token_swa[
                req.req_pool_idx,
                req.swa_alloc_offset : new_evicted_seqlen_swa,
            ]
            req.swa_alloc_offset = new_evicted_seqlen_swa
            self.token_to_kv_pool_allocator.free_swa(free_slots_swa)

        if new_evicted_seqlen_c4_state > req.c4_alloc_offset:
            free_slots_c4_state = self.req_to_token_pool.req_to_token_c4_state[
                req.req_pool_idx,
                req.c4_alloc_offset : new_evicted_seqlen_c4_state,
            ]
            req.c4_alloc_offset = new_evicted_seqlen_c4_state
            self.token_to_kv_pool_allocator.free_compress_state(
                free_slots_c4_state, "c4"
            )

        if new_evicted_seqlen_c128_state > req.c128_alloc_offset:
            free_slots_c128_state = self.req_to_token_pool.req_to_token_c128_state[
                req.req_pool_idx,
                req.c128_alloc_offset : new_evicted_seqlen_c128_state,
            ]
            req.c128_alloc_offset = new_evicted_seqlen_c128_state
            self.token_to_kv_pool_allocator.free_compress_state(
                free_slots_c128_state, "c128"
            )

    def supports_swa_c4c128(self) -> bool:
        return True

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        (
            kv_committed_len,
            _,
            _,
        ) = req.pop_committed_swa_c4_c128_kv_cache()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]
        swa_kv_indices = self.req_to_token_pool.req_to_token_swa[
            req.req_pool_idx, req.swa_alloc_offset : kv_committed_len
        ]
        c4_kv_indices = self.req_to_token_pool.req_to_token_c4[
            req.req_pool_idx, : kv_committed_len // 4
        ]
        c128_kv_indices = self.req_to_token_pool.req_to_token_c128[
            req.req_pool_idx, : kv_committed_len // 128
        ]
        c4_state_kv_indices = self.req_to_token_pool.req_to_token_c4_state[
            req.req_pool_idx, req.c4_alloc_offset : kv_committed_len
        ]
        c128_state_kv_indices = self.req_to_token_pool.req_to_token_c128_state[
            req.req_pool_idx, req.c128_alloc_offset : kv_committed_len
        ]

        self.token_to_kv_pool_allocator.swa_c4_c128_free(
            kv_indices,
            swa_kv_indices,
            c4_kv_indices,
            c128_kv_indices,
            c4_state_kv_indices,
            c128_state_kv_indices,
        )

    # TODO(zyj): prefix_cache
    def cache_unfinished_req(self, req: Req, chunked=False):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        swa_kv_indices = self.req_to_token_pool.req_to_token_swa[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        c4_kv_indices = self.req_to_token_pool.req_to_token_c4[
            req.req_pool_idx, : len(req.fill_ids) // 4
        ]
        c128_kv_indices = self.req_to_token_pool.req_to_token_c128[
            req.req_pool_idx, len(req.fill_ids) // 128
        ]
        c4_state_kv_indices = self.req_to_token_pool.req_to_token_c4_state[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        c128_state_kv_indices = self.req_to_token_pool.req_to_token_c128_state[
            req.req_pool_idx, : len(req.fill_ids)
        ]

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = [
            kv_indices.to(dtype=torch.int64, copy=True),
            swa_kv_indices.to(dtype=torch.int64, copy=True),
            c4_kv_indices.to(dtype=torch.int64, copy=True),
            c128_kv_indices.to(dtype=torch.int64, copy=True),
            c4_state_kv_indices.to(dtype=torch.int64, copy=True),
            c128_state_kv_indices.to(dtype=torch.int64, copy=True),
        ]

    def available_and_evictable_str(self) -> str:
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        c4_available_size = self.token_to_kv_pool_allocator.c4_available_size()
        c128_available_size = self.token_to_kv_pool_allocator.c128_available_size()
        c4_state_available_size = self.token_to_kv_pool_allocator.c4_state_available_size()
        c128_state_available_size = (
            self.token_to_kv_pool_allocator.c128_state_available_size()
        )
        return (
            f"Available swa tokens: {swa_available_size}\n"
            f"Available c4 tokens: {c4_available_size}\n"
            f"Available c128 tokens: {c128_available_size}\n"
            f"Available c4 state tokens: {c4_state_available_size}\n"
            f"Available c128 state tokens: {c128_state_available_size}\n"
        )