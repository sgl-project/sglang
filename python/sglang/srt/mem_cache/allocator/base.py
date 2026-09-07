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

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.utils.invariants import InvariantCheckLevel, resolve_level

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


def invariant_checks_enabled() -> bool:
    """Whether a side should build the tensors for its release-contract checks;
    `expect` gates only after its argument has been computed."""
    return resolve_level() >= InvariantCheckLevel.WARN


def iter_page_disjoint(segments, page_size: int):
    """Yield the non-empty ``(free_index, start_pos)`` segments of one kv row,
    asserting that consecutive segments never share a page."""
    prev_end = None
    for free_index, start_pos in segments:
        n = free_index.numel()
        if n == 0:
            continue
        assert (
            prev_end is None or start_pos // page_size > (prev_end - 1) // page_size
        ), f"segment at {start_pos} shares a page with the one ending at {prev_end}"
        prev_end = start_pos + n
        yield free_index, start_pos


class BaseKVPoolSide(abc.ABC):
    """One component of an allocator as the release and capacity paths see it.
    Every call takes full-attention slot ids; a side for another component
    translates them itself.

    Release contract: every id handed to ``free`` / ``free_segment`` is live on
    this side. Liveness is the caller's knowledge (the cache tombstoned the swa
    side, the ratchet moved the window), not something a side re-derives at
    free time, so no free path filters ids.
    """

    page_size: int
    # None: free right away. A list: hold frees until free_group_end().
    free_group: list[torch.Tensor] | None
    # The slot owner behind this side; the idle-time census reads its free list.
    pool: BaseKVPool

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
        for free_index, start_pos in iter_page_disjoint(segments, self.page_size):
            self.free_segment(free_index, start_pos=start_pos)

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


class BaseKVPool(abc.ABC):
    """Owns slots: hands them out and takes them back. The scheduler never
    holds a pool directly; an allocator wraps it and exposes it as a side."""

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

    # -- capacity --

    @abc.abstractmethod
    def available_size(self) -> int:
        raise NotImplementedError()

    def conserve_available_size(self) -> int:
        return self.available_size()

    def schedulable_available_size(self) -> int:
        return self.available_size()

    def get_all_free_pages(self):
        """Page free list for the idle-time census; None when the pool keeps
        none (watermark pools)."""
        return None

    # -- alloc --

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged pools")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged pools")

    # -- release --

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        """See `BaseKVPoolSide.free_segment`."""
        assert start_pos % self.page_size == 0, (
            f"segment start {start_pos} is not page-aligned"
        )
        self.free(free_index)

    def free_segments(self, segments):
        for free_index, start_pos in iter_page_disjoint(segments, self.page_size):
            self.free_segment(free_index, start_pos=start_pos)

    def free_group_begin(self):
        assert self.free_group is None, "free groups cannot be nested"
        self.free_group = []

    def free_group_end(self):
        pending, self.free_group = self.free_group, None
        if pending:
            self.free(torch.cat(pending))

    @staticmethod
    def _copy_for_free_group(free_index: torch.Tensor) -> torch.Tensor:
        return free_index.clone()

    # -- lifecycle / misc --

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    def resize(self, config) -> None:
        self.size = config.max_total_num_tokens
        if self.page_size > 1:
            self.num_pages = config.max_total_num_tokens // self.page_size
        self.clear()

    def get_kvcache(self):
        return self._kvcache

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

    def verify_byte_accounting(self) -> list:
        return []

    def debug_print(self) -> str:
        return ""


class BaseFreeListKVPool(BaseKVPool):
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


class KVPoolSide(BaseKVPoolSide):
    """A side backed by exactly one pool; every call forwards to it and the
    pool's own group defers the frees."""

    def __init__(self, pool: BaseKVPool):
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


class BaseKVAllocator(abc.ABC):
    """What the scheduler holds as `token_to_kv_pool_allocator`: the alloc
    entry points plus one `BaseKVPoolSide` per attention component in `sides`.
    Release and capacity go through a side, found by `ComponentType`; a free
    on the allocator itself fans out to every side, swa before full because
    the full side's peer check reads the pairing the swa side clears."""

    sides: dict[ComponentType, BaseKVPoolSide]
    size: int
    page_size: int
    dtype: torch.dtype
    device: str
    need_sort: bool

    def side(self, component_type: ComponentType) -> BaseKVPoolSide:
        try:
            return self.sides[component_type]
        except KeyError:
            raise KeyError(
                f"{type(self).__name__} has no {component_type} side; it has "
                f"{sorted(str(ct) for ct in self.sides)}"
            ) from None

    @property
    def full(self) -> BaseKVPoolSide:
        if ComponentType.FULL not in self.sides:
            raise AttributeError(f"{type(self).__name__} has no full side")
        return self.sides[ComponentType.FULL]

    @property
    def swa(self) -> BaseKVPoolSide:
        if ComponentType.SWA not in self.sides:
            raise AttributeError(f"{type(self).__name__} has no swa side")
        return self.sides[ComponentType.SWA]

    @property
    def size_full(self):
        return self.size

    # -- release: fan-out over the sides --

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        for side in self.sides.values():
            side.free(free_index)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        if free_index.numel() == 0:
            return
        for side in self.sides.values():
            side.free_segment(free_index, start_pos=start_pos)

    def free_segments(self, segments):
        for free_index, start_pos in iter_page_disjoint(segments, self.page_size):
            self.free_segment(free_index, start_pos=start_pos)

    def free_group_begin(self):
        for side in self.sides.values():
            side.free_group_begin()

    def free_group_end(self):
        for side in self.sides.values():
            side.free_group_end()

    # -- alloc --

    @abc.abstractmethod
    def available_size(self) -> int:
        """Slots the next alloc may take; the alloc pre-check."""
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocators")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocators")

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    def resize(self, config) -> None:
        raise NotImplementedError()

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
        """Page free list for the idle-time census; None for composites."""
        return None

    def translate_kv_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        return kv_indices

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError()

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        raise NotImplementedError()


class SinglePoolKVAllocator(BaseKVAllocator):
    """One pool behind one side; every entry point forwards to the pool."""

    def __init__(
        self, pool: BaseKVPool, *, component: ComponentType = ComponentType.FULL
    ):
        self.pool = pool
        self.sides = {component: KVPoolSide(pool)}
        self.page_size = pool.page_size
        self.dtype = pool.dtype
        self.device = pool.device
        self.need_sort = pool.need_sort
        self._kvcache = pool.get_kvcache()

    @property
    def size(self) -> int:
        return self.pool.size

    def available_size(self) -> int:
        return self.pool.available_size()

    def alloc(self, need_size: int):
        return self.pool.alloc(need_size)

    def alloc_extend(self, *args, **kwargs):
        return self.pool.alloc_extend(*args, **kwargs)

    def alloc_decode(self, *args, **kwargs):
        return self.pool.alloc_decode(*args, **kwargs)

    def clear(self):
        self.pool.clear()

    def resize(self, config) -> None:
        self.pool.resize(config)

    def get_all_free_pages(self):
        return self.pool.get_all_free_pages()

    def translate_kv_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.pool.translate_kv_indices_for_transfer(kv_indices)

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self.pool.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self.pool.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )

    def verify_byte_accounting(self) -> list:
        return self.pool.verify_byte_accounting()

    def debug_print(self) -> str:
        return self.pool.debug_print()
