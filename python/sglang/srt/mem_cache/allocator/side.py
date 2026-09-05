"""A pool side backed by exactly one allocator."""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.allocator.base import BaseKVAllocator, BaseKVPoolSide


class KVPoolSide(BaseKVPoolSide):
    """Forwards every side call to one pool; the pool's own group defers frees."""

    def __init__(self, pool: BaseKVAllocator):
        self.pool = pool
        self.page_size = pool.page_size
        self.free_group = None

    def available_size(self) -> int:
        return self.pool.available_size()

    def conserve_available_size(self) -> int:
        return self.pool.conserve_available_size()

    def schedulable_available_size(self) -> int:
        return self.pool.schedulable_available_size()

    def free(self, free_index: torch.Tensor):
        self.pool.free(free_index)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        self.pool.free_segment(free_index, start_pos=start_pos)

    def free_group_begin(self):
        self.pool.free_group_begin()

    def free_group_end(self):
        self.pool.free_group_end()
