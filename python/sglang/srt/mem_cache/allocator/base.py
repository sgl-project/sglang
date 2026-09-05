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

from sglang.srt.utils.invariants import InvariantCheckLevel, resolve_level

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


def invariant_checks_enabled() -> bool:
    """Whether a side should build the tensors for its release-contract checks;
    `expect` gates only after its argument has been computed."""
    return resolve_level() >= InvariantCheckLevel.WARN


class BaseKVPoolSide(abc.ABC):
    """One pool as the scheduler and the caches see it: capacity views plus the
    release path. A single-pool allocator is its own side; a hybrid allocator
    exposes one side per pool, every side addressed by full-attention slot ids.

    Release contract: every id handed to ``free`` / ``free_segment`` is live on
    this side. Liveness is the caller's knowledge (the cache tombstoned the swa
    side, the ratchet moved the window), not something a side re-derives from
    its pairing at free time, so no free path filters ids.
    """

    page_size: int
    # None: free right away. A list: hold frees until free_group_end().
    free_group: list[torch.Tensor] | None

    # -- capacity views --

    @abc.abstractmethod
    def available_size(self) -> int:
        """Slots a caller may allocate right now."""
        raise NotImplementedError()

    def conserve_available_size(self) -> int:
        """Slot-conservation view for the leak invariant: static capacity minus
        live slots. Equal to ``available_size`` unless bytes are shared."""
        return self.available_size()

    def schedulable_available_size(self) -> int:
        """Planner view: realizable with compaction. Equal to ``available_size``
        unless the pool compacts lazily."""
        return self.available_size()

    # -- release --

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        """Free ``kv_row[start_pos : start_pos + n]`` of one request.

        In page units the segment is ``[start_pos // ps, ceil(end / ps))``:
        ``start_pos`` sits on a page boundary, the end may fall mid-page, and
        the whole last page is released."""
        assert start_pos % self.page_size == 0, (
            f"segment start {start_pos} is not page-aligned"
        )
        self.free(free_index)

    def free_segments(self, segments):
        """Free several ``(free_index, start_pos)`` segments of one request's
        kv row. Each covers pages ``[start_pos // ps, ceil(end / ps))``; starts
        are page-aligned and consecutive page ranges do not overlap, so every
        page is released exactly once."""
        for free_index, start_pos in self._page_disjoint(segments):
            self.free_segment(free_index, start_pos=start_pos)

    def _page_disjoint(self, segments):
        ps = self.page_size
        prev_end = None
        for free_index, start_pos in segments:
            n = free_index.numel()
            if n == 0:
                continue
            assert prev_end is None or start_pos // ps > (prev_end - 1) // ps, (
                f"segment at {start_pos} shares a page with the one ending at {prev_end}"
            )
            prev_end = start_pos + n
            yield free_index, start_pos

    def free_group_begin(self):
        assert self.free_group is None, "free groups cannot be nested"
        self.free_group = []

    def free_group_end(self):
        pending, self.free_group = self.free_group, None
        if pending:
            self.free(torch.cat(pending))

    @staticmethod
    def _copy_for_free_group(free_index: torch.Tensor) -> torch.Tensor:
        """Take ownership before a caller can mutate a deferred tensor view."""
        return free_index.clone()


class BaseKVAllocator(BaseKVPoolSide):
    """Allocation plus the side contract, with no free-list state of its own:
    a pool that keeps a page free list derives from `BaseFreeListKVAllocator`,
    composites keep none."""

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
        self.free_group: list[torch.Tensor] | None = None

    @property
    def size_full(self):
        return self.size

    # -- scheduler-facing capacity hooks --
    # The scheduler calls these unconditionally, with no allocator-type branches
    # on its side; byte-accounted composites override the token-count defaults.

    def evict_to_free_tokens(self, tree_cache, num_tokens: int) -> None:
        """Evict unlocked prefix-cache entries until this allocator can serve
        ``num_tokens`` or nothing evictable remains."""
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        evict_from_tree_cache(tree_cache, num_tokens)

    def check_decode_capacity(self, *, num_tokens: int, tree_cache) -> bool:
        """Whether the next decode step's ``num_tokens`` allocation fits after
        evicting reclaimable cache. The retract loop converges on this same
        check, so a shortfall here retracts instead of failing in alloc."""
        self.evict_to_free_tokens(tree_cache, num_tokens)
        return self.available_size() >= num_tokens

    def verify_byte_accounting(self) -> list:
        """Idle-time diagnostic: recompute byte/slot accounting and return
        violation strings, empty when healthy. Static pools have no byte model."""
        return []

    def debug_print(self) -> str:
        return ""

    def get_kvcache(self):
        return self._kvcache

    def get_all_free_pages(self):
        """Page free list for the idle-time census; None when the pool keeps
        none (composites, watermark pools)."""
        return None

    def translate_kv_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """Token ids as the PD transfer engine addresses them. Identity here
        because a static pool's ids index its registered buffers directly;
        virtual-id pools must override."""
        return kv_indices

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
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


class BaseFreeListKVAllocator(BaseKVAllocator):
    """A pool whose free slots live in a page free list (`free_pages`, plus the
    unsorted `release_pages` staging area when `need_sort`)."""

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
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.free_pages = None
        self.release_pages = None

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_all_free_pages(self):
        if self.release_pages is None or len(self.release_pages) == 0:
            return self.free_pages
        return torch.cat((self.free_pages, self.release_pages))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )
