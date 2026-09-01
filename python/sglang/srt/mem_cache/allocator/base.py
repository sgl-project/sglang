"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        # None: free right away. A list: hold frees until free_group_end().
        self.free_group: list[torch.Tensor] | None = None

    @property
    def size_full(self):
        return self.size

    # -- scheduler-facing capacity hooks --
    # The scheduler calls these UNCONDITIONALLY (zero feature branches on its
    # side); the defaults reproduce the historical token behavior exactly, and
    # unified composites override them with byte-denominated logic.

    def evict_to_free_tokens(self, tree_cache, num_tokens: int) -> None:
        """Ask the prefix cache to evict unlocked entries until this allocator
        can serve ``num_tokens`` (or nothing evictable remains). Default = the
        shared token-count eviction; joint-byte composites override (evicting
        one multi-lifetime tree node frees bytes on several sides at once).
        """
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        evict_from_tree_cache(tree_cache, num_tokens)

    def check_decode_capacity(self, *, num_tokens: int, tree_cache) -> bool:
        """Whether the NEXT decode step's ``num_tokens`` allocation fits,
        evicting reclaimable cache first. The retract loop converges on this
        same check, so allocator-side shortfalls retract gracefully instead of
        tripping fail-loud alloc errors. Default reproduces the historical
        ``ScheduleBatch.check_decode_mem`` body; unified composites override
        with byte gates + per-step reservations of their own.
        """
        self.evict_to_free_tokens(tree_cache, num_tokens)
        return self.available_size() >= num_tokens

    def verify_byte_accounting(self) -> list:
        """Idle-time conservation diagnostic: recompute this allocator's
        byte/slot accounting and return human-readable violation strings
        (empty == healthy). Default: static pools have no byte model.
        """
        return []

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def free_group_begin(self):
        self.free_group = []

    def free_group_end(self):
        pending, self.free_group = self.free_group, None
        if pending:
            self.free(torch.cat(pending))

    @staticmethod
    def _copy_for_free_group(free_index: torch.Tensor) -> torch.Tensor:
        """Take ownership before a caller can mutate a deferred tensor view."""
        return free_index.clone()

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def translate_kv_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """Token ids as the PD-disaggregation transfer engine addresses them.

        Identity here: a static pool's token ids index its registered buffers
        directly. Virtual-id pools must override.
        """
        return kv_indices

    def get_cpu_copy(self, indices, mamba_indices=None):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def resize(self, config) -> None:
        self.size = config.max_total_num_tokens
        if self.page_size > 1:
            self.num_pages = config.max_total_num_tokens // self.page_size
        self.clear()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()

    def free_full(self, free_index: torch.Tensor):
        """Free slots whose SWA peers the caller already released.

        A hybrid SWA allocator pairs each full-attention slot with an SWA slot
        that can die first; this releases the full side alone. A single pool has
        no peer, so it is a plain free()."""
        self.free(free_index)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        """Free ``kv_row[start_pos : start_pos + n]`` of one request (or a
        page-aligned copy); subclasses may use ``start_pos`` to skip the
        data-dependent dedup. Default: plain free()."""
        self.free(free_index)

    def free_segments(self, segments):
        """Free disjoint ascending ``(free_index, start_pos)`` segments of one
        request's kv row; a boundary page shared by consecutive segments is
        emitted once (the later segment's head is trimmed)."""
        ps = self.page_size
        prev_end = None
        for free_index, start_pos in segments:
            n = free_index.numel()
            if n == 0:
                continue
            seg_end = start_pos + n
            if prev_end is not None and start_pos // ps == (prev_end - 1) // ps:
                boundary = (start_pos // ps + 1) * ps
                free_index = free_index[boundary - start_pos :]
                start_pos = boundary
            prev_end = seg_end
            self.free_segment(free_index, start_pos=start_pos)
