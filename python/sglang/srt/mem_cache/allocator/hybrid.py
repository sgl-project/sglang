"""Hybrid SWA allocators: two pools paired by full-attention slot id."""

from __future__ import annotations

import abc

import torch

from sglang.srt.mem_cache.allocator.base import BaseKVAllocator
from sglang.srt.mem_cache.allocator.pairing import BaseFullToSWAPairing


class BaseHybridSWAKVAllocator(BaseKVAllocator):
    """A full-attention pool and an SWA pool addressed by FULL slot ids.

    ``sides`` holds the swa side before the full side, so a fan-out free
    clears the pairing before the full side's peer check reads it; ``pairing``
    maps a full id to its swa peer. alloc stays with each implementation: the
    two sides consume one shared page count, so it cannot be split per side."""

    pairing: BaseFullToSWAPairing

    @property
    @abc.abstractmethod
    def size_swa(self) -> int:
        raise NotImplementedError()

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        return self.pairing.translate(kv_indices)

    def available_size(self) -> int:
        return min(side.available_size() for side in self.sides.values())
