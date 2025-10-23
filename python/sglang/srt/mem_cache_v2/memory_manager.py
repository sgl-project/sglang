from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.mem_cache_v2.cache_index import (
    CacheIndex,
    KeyFunc_t,
    MatchResult,
    ReqPool,
)
from sglang.srt.mem_cache_v2.memory_pool import BufferInfo, MemoryPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@dataclass(frozen=True)
class CacheView:
    """Used by attention backend to access the cache."""

    # attention backend is expected to index on this tensor as cache_view.req_to_token[req_pool_idx, :]
    req_to_token: (
        torch.Tensor
    )  # this req_to_token pool is on gpu and will be populated in model_runner.forward_* or cuda_graph_runner.replay_prepare
    memory_pool: Any
    ready_event: Any  # event to wait for the transfer of this view to be ready


class Allocator:
    def __init__(self, size: int, page_size: int):
        self.size = size
        self.page_size = page_size
        self.page_num = size // page_size
        # page 0 is reserved for dummy write
        self.free_list = list(range(1, self.page_num + 1))

    def alloc(self, num_needed: int, last_loc: int = 0) -> list[int]:
        """Allocate num_needed tokens. last_loc=0 means fresh allocation."""
        if num_needed == 0:
            return []

        allocated = []
        remaining = num_needed

        # Fill partial page
        if last_loc > 0 and (last_loc + 1) % self.page_size != 0:
            next_idx = last_loc + 1
            slots_left = self.page_size - (next_idx % self.page_size)
            fill = min(remaining, slots_left)
            allocated.extend(range(next_idx, next_idx + fill))
            remaining -= fill

        # Allocate new pages
        if remaining > 0:
            num_pages = (remaining + self.page_size - 1) // self.page_size
            if num_pages > len(self.free_list):
                raise RuntimeError(
                    f"Out of memory: cannot allocate {num_needed} tokens as {num_pages} pages are needed"
                )

            pages = self.free_list[:num_pages]
            self.free_list = self.free_list[num_pages:]

            # Full pages take page_size, last page takes what's left
            full_pages = pages[:-1]
            last_page = pages[-1]
            last_page_size = remaining - len(full_pages) * self.page_size

            allocated.extend(
                [
                    page * self.page_size + offset
                    for page in full_pages
                    for offset in range(self.page_size)
                ]
            )
            allocated.extend(
                [
                    last_page * self.page_size + offset
                    for offset in range(last_page_size)
                ]
            )

        return allocated

    def free(self, indices: list[int]):
        page_indices = set(index // self.page_size for index in indices)
        self.free_list.extend(page_indices)


@dataclass(frozen=True)
class AllocationResult:
    allocated_indices: torch.Tensor
    req_pool_idx: int


@dataclass
class ReqInfo:
    allocation_key: Any
    last_loc: int
    allocated_len: int
    req_pool_idx: int
    init_cached_len: int  # Initial matched prefix from allocation, never changes
    cached_len: int  # Updated after each update_cache call


class MemoryManager:
    def __init__(
        self,
        cache_index: CacheIndex,
        memory_pools: dict[tuple[int, ...], MemoryPool],
        page_size: int,
        size: int,
        max_ctx: int,
        device: str,
    ):
        self.cache_index = cache_index
        self.memory_pools = memory_pools
        self.page_size = page_size
        self.size = size
        self.page_num = size // page_size
        self.device = device

        assert size % page_size == 0, "Size should be a multiple of page size"

        self.req_info: dict[str, ReqInfo] = {}

        # Ensure req_pool has at least 16 slots, up to 2048
        req_pool_size = max(128, min(2048, (size // max_ctx) * 8))
        self.req_pool = ReqPool(req_pool_size, max_ctx, device)
        self.allocator = Allocator(size, page_size)

    # Allocation
    def allocate_request(
        self, req: Req, include_last: bool, match_result: MatchResult
    ) -> AllocationResult:
        """
        This function will allocate the memory for the request.
        If include_last is True, req.seqlen tokens will be allocated. (e.g. prefill)
        If include_last is False, the last output token will not be allocated. (e.g. PD, retraction)
        """
        if include_last:
            token_ids = req.origin_input_ids + req.output_ids
        else:
            token_ids = req.origin_input_ids + (
                req.output_ids[:-1] if req.output_ids else []
            )
        # 1. lock the index
        not_ready_indices = self.cache_index.allocate(match_result.allocation_key)
        # TODO: start cpu -> gpu loading
        cached_indices = torch.cat([match_result.matched_indices, not_ready_indices])

        # 2. allocate unmatched tokens
        miss_len = len(token_ids) - len(cached_indices)
        last_loc = int(cached_indices[-1].item()) if len(cached_indices) > 0 else 0
        new_indices = self.allocator.alloc(miss_len, last_loc=last_loc)
        allocated_indices = torch.cat(
            [cached_indices, torch.tensor(new_indices, dtype=torch.int32)]
        )
        # 3. update per-request allocation info
        req_pool_idx = self.req_pool.alloc()
        cached_len = len(match_result.matched_indices)
        self.req_info[req.rid] = ReqInfo(
            allocation_key=match_result.allocation_key,
            last_loc=int(allocated_indices[-1].item()),
            allocated_len=len(allocated_indices),
            req_pool_idx=req_pool_idx,
            init_cached_len=cached_len,
            cached_len=cached_len,
        )
        self.req_pool.cpu_pool[req_pool_idx, : len(allocated_indices)] = (
            allocated_indices
        )

        return AllocationResult(
            allocated_indices=allocated_indices.to(self.device),
            req_pool_idx=req_pool_idx,
        )

    def allocate_tokens(self, req: Req, num_token: int) -> AllocationResult:
        """
        This function is called to allocate a number of tokens.
        The typical usage is to allocate for decode.
        """
        req_info = self.req_info[req.rid]
        new_indices = self.allocator.alloc(num_token, last_loc=req_info.last_loc)
        if not new_indices:
            raise RuntimeError(f"Out of memory: cannot allocate {num_token} tokens")
        new_indices_tensor = torch.tensor(new_indices, dtype=torch.int32)
        self.req_pool.cpu_pool[
            req_info.req_pool_idx,
            req_info.allocated_len : req_info.allocated_len + num_token,
        ] = new_indices_tensor
        req_info.allocated_len += num_token
        req_info.last_loc = new_indices[-1]
        return AllocationResult(
            allocated_indices=new_indices_tensor.to(self.device),
            req_pool_idx=req_info.req_pool_idx,
        )

    # Free memory and update index
    def update_cache(self, req: Req):
        """
        Update the cache index for the request.
        Frees duplicate indices, updates req_pool with tree indices, and updates allocation_key.
        """
        req_info = self.req_info[req.rid]
        token_ids = req.origin_input_ids + (
            req.output_ids[:-1] if req.output_ids else []
        )
        assert len(token_ids) == req_info.allocated_len, "Allocated length mismatch"

        aligned_len = (len(token_ids) // self.page_size) * self.page_size
        key = tuple(token_ids[:aligned_len])
        values = self.req_pool.cpu_pool[req_info.req_pool_idx, :aligned_len]

        allocation_key, matched_indices = self.cache_index.insert(
            key, values, req_info.allocation_key
        )
        matched_len = len(matched_indices)

        # Free duplicates: from last cached to newly matched portion
        # These are indices we allocated but are now replaced by tree indices
        overlapped = values[req_info.cached_len : matched_len]
        if len(overlapped) > 0:
            self.allocator.free(overlapped.tolist())

        # Update req_pool with tree indices (matched_indices are what's in tree before this insert)
        self.req_pool.cpu_pool[
            req_info.req_pool_idx, req_info.cached_len : matched_len
        ] = matched_indices[req_info.cached_len : matched_len]

        req_info.allocation_key = allocation_key
        # After insert, aligned_len tokens are now in the tree
        req_info.cached_len = aligned_len

    def release_req(self, req: Req):
        """
        Release the request from the cache.
        """
        req_info = self.req_info.pop(req.rid)
        unlocked = self.cache_index.free(req_info.allocation_key)
        assert unlocked == req_info.cached_len, "Unlocked length mismatch"

        uninserted = self.req_pool.cpu_pool[
            req_info.req_pool_idx, unlocked : req_info.allocated_len
        ]
        self.allocator.free(uninserted.tolist())
        self.req_pool.free([req_info.req_pool_idx])

    def evict(self, num_tokens: int):
        evicted = self.cache_index.evict(num_tokens)
        self.allocator.free(evicted.tolist())
        return len(evicted)

    # Query info & attribute
    def match_prefix(
        self, token_ids: list[int], key_func: KeyFunc_t | None = None
    ) -> MatchResult:
        """
        Match the prefix of the request from the cache.
        """
        if key_func:
            key = key_func(token_ids)
        else:
            key = tuple(token_ids)
        return self.cache_index.match_prefix(key)

    @lru_cache()
    def get_buf_infos(self) -> list[BufferInfo]:
        """
        Get the contiguous buffer infos from the cache.
        """
        buf_infos = []
        for memory_pool in self.memory_pools.values():
            buf_infos.extend(memory_pool.get_buf_info())
        return buf_infos

    @lru_cache(maxsize=128)
    def get_cache_view(self, layer_id: int) -> CacheView:
        for layers, pool in self.memory_pools.items():
            if layer_id in layers:
                return CacheView(
                    req_to_token=self.req_pool.cpu_pool,
                    memory_pool=pool,
                    ready_event=None,  # for cpu -> gpu loading
                )
        raise ValueError(f"Layer {layer_id} not found in memory pools")
