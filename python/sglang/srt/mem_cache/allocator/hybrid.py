"""Hybrid SWA allocators: two pools paired by full-attention slot id."""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.allocator.base import BaseKVAllocator, KVFreeSide
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


class BaseHybridSWAKVAllocator(BaseKVAllocator):
    """A full-attention pool and an SWA pool addressed by FULL slot ids.

    ``full`` and ``swa`` are the two sides; both take full slot ids, the swa
    side translates them to its own pool. Combined frees fan out to both
    sides. alloc stays with each implementation: the two sides consume one
    shared page count, so it cannot be split per side."""

    full: KVFreeSide
    swa: KVFreeSide

    def side(self, component_type: ComponentType) -> KVFreeSide:
        if component_type == ComponentType.FULL:
            return self.full
        if component_type == ComponentType.SWA:
            return self.swa
        raise KeyError(f"hybrid SWA allocator has no {component_type} side")

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        # SWA first: it reads the pairing, which a later cache action in this
        # group may re-point at another SWA slot.
        self.swa.free(free_index)
        self.full.free(free_index)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        if free_index.numel() == 0:
            return
        self.swa.free_segment(free_index, start_pos=start_pos)
        self.full.free_segment(free_index, start_pos=start_pos)

    def free_group_begin(self):
        super().free_group_begin()
        self.swa.free_group_begin()
        self.full.free_group_begin()

    def free_group_end(self):
        super().free_group_end()
        self.swa.free_group_end()
        self.full.free_group_end()

    def available_size(self):
        return min(self.full.available_size(), self.swa.available_size())
