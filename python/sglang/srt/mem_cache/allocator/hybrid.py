"""Hybrid SWA allocators: two pools paired by full-attention slot id."""

from __future__ import annotations

import abc

import torch

from sglang.srt.mem_cache.allocator.base import BaseKVAllocator, BaseKVPoolSide
from sglang.srt.mem_cache.allocator.pairing import BaseFullToSWAPairing


class BaseHybridSWAKVAllocator(BaseKVAllocator):
    """A full-attention pool and an SWA pool addressed by FULL slot ids.

    ``full`` and ``swa`` are the two sides and ``pairing`` maps a full id to its
    swa peer. A free on the allocator itself fans out to both sides. alloc stays
    with each implementation: the two sides consume one shared page count, so
    it cannot be split per side."""

    full: BaseKVPoolSide
    swa: BaseKVPoolSide
    pairing: BaseFullToSWAPairing

    @property
    @abc.abstractmethod
    def size_swa(self) -> int:
        raise NotImplementedError()

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return self.pairing.translate(kv_indices)

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

    # No group of its own: each side defers its half.
    def free_group_begin(self):
        self.swa.free_group_begin()
        self.full.free_group_begin()

    def free_group_end(self):
        self.swa.free_group_end()
        self.full.free_group_end()

    def available_size(self) -> int:
        return min(self.full.available_size(), self.swa.available_size())

    def conserve_available_size(self) -> int:
        return min(
            self.full.conserve_available_size(), self.swa.conserve_available_size()
        )

    def schedulable_available_size(self) -> int:
        return min(
            self.full.schedulable_available_size(),
            self.swa.schedulable_available_size(),
        )
