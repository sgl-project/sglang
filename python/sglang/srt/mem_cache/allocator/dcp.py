from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class DcpTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Logical paged allocator for decode-context-parallel KV storage.

    DCP keeps request metadata in a cluster-wide logical token-index space.
    KV writers map logical token index ``i`` to rank ``i % dcp_world_size`` and
    local physical offset ``i // dcp_world_size`` when writing the per-rank pool.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
        dcp_rank: int,
        dcp_world_size: int,
    ):
        assert dcp_world_size >= 1
        assert size % dcp_world_size == 0, (
            f"DCP logical size {size} must be divisible by dcp_world_size "
            f"{dcp_world_size}"
        )
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort
        self.dcp_rank = dcp_rank
        self.dcp_world_size = dcp_world_size
        self.real_allocator = PagedTokenToKVPoolAllocator(
            size, page_size, dtype, device, kvcache, need_sort
        )

    @property
    def free_pages(self):
        return self.real_allocator.free_pages

    @free_pages.setter
    def free_pages(self, value):
        self.real_allocator.free_pages = value

    @property
    def release_pages(self):
        return self.real_allocator.release_pages

    @release_pages.setter
    def release_pages(self, value):
        self.real_allocator.release_pages = value

    @property
    def is_not_in_free_group(self):
        return self.real_allocator.is_not_in_free_group

    @is_not_in_free_group.setter
    def is_not_in_free_group(self, value):
        self.real_allocator.is_not_in_free_group = value

    @property
    def free_group(self):
        return self.real_allocator.free_group

    @free_group.setter
    def free_group(self, value):
        self.real_allocator.free_group = value

    @property
    def num_pages(self):
        return self.real_allocator.num_pages

    @property
    def size_full(self):
        return self.size

    def alloc(self, need_size: int):
        return self.real_allocator.alloc(need_size)

    def alloc_extend(self, *args, **kwargs):
        return self.real_allocator.alloc_extend(*args, **kwargs)

    def alloc_decode(self, *args, **kwargs):
        return self.real_allocator.alloc_decode(*args, **kwargs)

    def free(self, free_index: torch.Tensor):
        return self.real_allocator.free(free_index)

    def clear(self):
        return self.real_allocator.clear()

    def available_size(self):
        return self.real_allocator.available_size()

    def debug_print(self) -> str:
        return self.real_allocator.debug_print()

    def free_group_begin(self):
        self.real_allocator.free_group_begin()

    def free_group_end(self):
        self.real_allocator.free_group_end()

    def merge_and_sort_free(self):
        self.real_allocator.merge_and_sort_free()

    def backup_state(self):
        return self.real_allocator.backup_state()

    def restore_state(self, state):
        self.real_allocator.restore_state(state)

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self.real_allocator.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self.real_allocator.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )

    def filter_local_indices(self, indices: torch.Tensor) -> torch.Tensor:
        local_mask = indices % self.dcp_world_size == self.dcp_rank
        return indices[local_mask] // self.dcp_world_size
