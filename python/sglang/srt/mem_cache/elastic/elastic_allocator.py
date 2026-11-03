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

import logging
import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import (
    ElasticAllocator,
    ElasticMempoolOrchestrator,
    cu_page_size,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.elastic.elastic_memory_pool import (
        ElasticMHATokenToKVPool,
    )
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

logger = logging.getLogger(__name__)


class ElasticTokenToKVPoolAllocator(TokenToKVPoolAllocator, ElasticAllocator):
    """page_size=1 allocator over an `ElasticMHATokenToKVPool` that can shrink
    its tail or grow at the end.

    While a shrink is pending, slots above `candidate_size` are withheld: free
    ones sit in `candidate_unmap_pages` (still counted as available), and the
    shrink happens once every in-use tail slot is evictable from the tree.
    A shrink never goes below `min_size` slots.
    """

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: ElasticMHATokenToKVPool,
        need_sort: bool,
        max_size: Optional[int] = None,
    ):
        super().__init__(size, dtype, device, kvcache, need_sort)
        # Smallest shrink worth an unmap: two cu_pages of this pool's slots.
        self.cu_page_token_num = cu_page_size() // kvcache.state_memsize + 1
        # Grow cap in slots (bounded by the full->SWA mapping for the full pool).
        self.max_size = max_size
        self.min_size = 0
        self.map_threshold = envs.SGLANG_ELASTIC_MEM_MAP_THRESHOLD.get()
        self.unmap_threshold = envs.SGLANG_ELASTIC_MEM_UNMAP_THRESHOLD.get()
        self._evictable_size_fn: Callable[[], int] = lambda: 0
        self._evict_fn: Callable[[int], None] = lambda num_tokens: None
        self._evictable_indices_fn: Callable[[], torch.Tensor] = self._no_indices

    @property
    def pool_name(self) -> str:
        return self._kvcache.allocation_label or "kv"

    def bind_tree_cache_fns(
        self,
        *,
        evictable_size_fn: Callable[[], int],
        evict_fn: Callable[[int], None],
        evictable_indices_fn: Callable[[], torch.Tensor],
    ) -> None:
        self._evictable_size_fn = evictable_size_fn
        self._evict_fn = evict_fn
        self._evictable_indices_fn = evictable_indices_fn

    def set_min_size(self, min_size: int) -> None:
        self.min_size = min(min_size, self.size)

    def clear(self):
        super().clear()
        # Slot -> handed out by alloc and not yet freed (tree-cached slots stay True).
        self.in_use = torch.zeros(
            (self.size + 1,), dtype=torch.bool, device=self.device
        )
        self.candidate_size = self.size
        self.candidate_unmap_pages = self._empty_pages()
        self._failed_drain_key: Optional[Tuple[int, int]] = None

    def available_size(self):
        return (
            len(self.free_pages)
            + len(self.release_pages)
            + len(self.candidate_unmap_pages)
        )

    def get_all_free_pages(self):
        return torch.cat(
            (self.free_pages, self.release_pages, self.candidate_unmap_pages)
        )

    def alloc(self, need_size: int):
        self._borrow_candidate_pages(need_size)
        select_index = super().alloc(need_size)
        if select_index is not None:
            self.in_use[select_index] = True
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.free_group is not None:
            # Deferred; free_group_end re-enters free() with the group closed.
            super().free(free_index)
            return

        self.in_use[free_index] = False
        if self._shrink_pending():
            tail_mask = free_index > self.candidate_size
            self.candidate_unmap_pages = torch.cat(
                (self.candidate_unmap_pages, free_index[tail_mask])
            )
            free_index = free_index[~tail_mask]
        super().free(free_index)

    # ---- ElasticAllocator ----

    def can_be_candidate(self) -> bool:
        return True

    def token_usage(self) -> float:
        num_used = self.size - (self.available_size() + self._evictable_size_fn())
        return num_used / self.size

    def can_unmap(self) -> bool:
        return self.token_usage() < self.unmap_threshold

    def can_map(self) -> bool:
        if self.max_size is not None and self.size >= self.max_size:
            return False
        return self.token_usage() > self.map_threshold

    def mark_unmap_candidate(
        self, is_candidate: bool
    ) -> Optional[ElasticTokenToKVPoolAllocator]:
        if not is_candidate:
            self._cancel_shrink()
            return None

        assert not self._shrink_pending(), (
            f"{self.pool_name}: {self.candidate_size=}, {self.size=}"
        )
        new_size = max(int((self.token_usage() + 1) / 2 * self.size), self.min_size)
        if self.size - new_size < 2 * self.cu_page_token_num:
            return None

        self.candidate_size = new_size
        self.candidate_unmap_pages = self._take_tail(self.candidate_size)
        self._failed_drain_key = None
        logger.info(
            f"elasticmem: mark {self.pool_name} unmap candidate, "
            f"{self.size=}, {self.candidate_size=}"
        )
        return self

    def can_do_unmap(self) -> bool:
        tail_in_use = self._tail_in_use_indices()
        if tail_in_use.numel() == 0:
            return True
        evictable_size = self._evictable_size_fn()
        if tail_in_use.numel() > evictable_size:
            return False
        # The evictable-indices lookup walks the whole tree; skip it until the
        # tail or the evictable total changes since the last failed check.
        drain_key = (tail_in_use.numel(), evictable_size)
        if drain_key == self._failed_drain_key:
            return False
        evictable = self._evictable_indices_fn()
        drainable = bool(torch.isin(tail_in_use, evictable).all())
        self._failed_drain_key = None if drainable else drain_key
        return drainable

    def reduce(self) -> int:
        start_time = time.perf_counter()
        self._evict_tail()
        if len(self.candidate_unmap_pages) != self.size - self.candidate_size:
            # A tail slot got locked between can_do_unmap and now; retry later.
            logger.warning(
                f"elasticmem: {self.pool_name} tail not drained, skip reduce "
                f"({len(self.candidate_unmap_pages)=}, {self.size=}, "
                f"{self.candidate_size=})"
            )
            return 0

        # Unmap is not stream-ordered: with overlap scheduling the in-flight
        # forward may still read freed or just-evicted tail slots.
        torch.get_device_module(self.device).synchronize()
        unmap_num, cur_size = self._kvcache.reduce(self.candidate_size)
        # cu_page rounding can leave slots above candidate_size backed; free them.
        kept = self.candidate_unmap_pages[self.candidate_unmap_pages <= cur_size]
        self.candidate_unmap_pages = self._empty_pages()
        self.in_use = self.in_use[: cur_size + 1]
        self.size = cur_size
        self.candidate_size = cur_size
        super().free(kept)

        logger.info(
            f"elasticmem: {self.pool_name} reduce to {cur_size} took "
            f"{(time.perf_counter() - start_time) * 1000:.2f} ms"
        )
        return unmap_num

    def expand(self, expand_size: int) -> int:
        start_time = time.perf_counter()
        assert not self._shrink_pending()

        if self.max_size is not None:
            expand_size = min(expand_size, self.max_size - self.size)
        if expand_size <= 0:
            return 0

        map_num, cur_size = self._kvcache.expand(self.size + expand_size)
        new_pages = torch.arange(
            self.size + 1, cur_size + 1, dtype=torch.int64, device=self.device
        )
        self.in_use = torch.cat(
            (
                self.in_use,
                torch.zeros(
                    (cur_size - self.size,), dtype=torch.bool, device=self.device
                ),
            )
        )
        self.size = cur_size
        self.candidate_size = cur_size
        super().free(new_pages)

        logger.info(
            f"elasticmem: {self.pool_name} expand to {cur_size} took "
            f"{(time.perf_counter() - start_time) * 1000:.2f} ms"
        )
        return map_num

    def cu_page_to_token(self, cu_page_num: int) -> int:
        return self._kvcache.cu_page_to_token(cu_page_num)

    def update_size(self) -> None:
        assert self.size == self._kvcache.size

    # ---- helpers ----

    def _empty_pages(self) -> torch.Tensor:
        return torch.empty((0,), dtype=torch.int64, device=self.device)

    def _no_indices(self) -> torch.Tensor:
        return self._empty_pages()

    def _shrink_pending(self) -> bool:
        return self.candidate_size < self.size

    def _take_tail(self, boundary: int) -> torch.Tensor:
        """Remove and return the free slots above boundary."""
        release_tail = self.release_pages > boundary
        free_tail = self.free_pages > boundary
        tail = torch.cat((self.release_pages[release_tail], self.free_pages[free_tail]))
        self.release_pages = self.release_pages[~release_tail]
        self.free_pages = self.free_pages[~free_tail]
        return tail

    def _return_candidate_pages(self, pages: torch.Tensor) -> None:
        if self.need_sort:
            self.release_pages = torch.cat((self.release_pages, pages))
        else:
            self.free_pages = torch.cat((self.free_pages, pages))

    def _borrow_candidate_pages(self, need_size: int) -> None:
        """Lend withheld tail slots to an alloc the free lists cannot cover.

        The scheduler sized its eviction against available_size(), which
        counts them; the borrowed slots just keep the shrink pending longer.
        """
        if not self._shrink_pending():
            return
        shortfall = need_size - (len(self.free_pages) + len(self.release_pages))
        if shortfall <= 0:
            return
        self._return_candidate_pages(self.candidate_unmap_pages[:shortfall])
        self.candidate_unmap_pages = self.candidate_unmap_pages[shortfall:]

    def _cancel_shrink(self) -> None:
        if not self._shrink_pending():
            return
        self._return_candidate_pages(self.candidate_unmap_pages)
        self.candidate_unmap_pages = self._empty_pages()
        self.candidate_size = self.size
        self._failed_drain_key = None
        logger.info(f"elasticmem: cancel {self.pool_name} shrink, {self.size=}")

    def _tail_in_use_indices(self) -> torch.Tensor:
        tail = self.in_use[self.candidate_size + 1 :]
        return tail.nonzero(as_tuple=True)[0] + (self.candidate_size + 1)

    def _evict_tail(self) -> None:
        """LRU-evict from the tree until no in-use slot remains in the tail.

        The tree cannot evict specific slots, so this may evict colder entries
        below the tail too; can_do_unmap guarantees the tail itself is evictable.
        """
        start_time = time.perf_counter()
        while True:
            num_tail_in_use = int(self.in_use[self.candidate_size + 1 :].sum())
            evictable = self._evictable_size_fn()
            if num_tail_in_use == 0 or evictable == 0:
                break
            self._evict_fn(num_tail_in_use)
            if self._evictable_size_fn() == evictable:
                break
        logger.debug(
            f"elasticmem: {self.pool_name} evict_tail took "
            f"{(time.perf_counter() - start_time) * 1000:.2f} ms"
        )


class ElasticSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator, ElasticAllocator):
    """SWA allocator whose full and SWA sub-allocators trade physical pages
    through an owned `ElasticMempoolOrchestrator`."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: SWAKVPool,
        need_sort: bool,
        req_to_token_pool=None,
    ):
        assert page_size == 1, "elasticmem supports page_size == 1 only"
        super().__init__(
            size,
            size_swa,
            page_size,
            dtype,
            device,
            kvcache,
            need_sort,
            req_to_token_pool=req_to_token_pool,
        )
        assert not self._swa_req_ring, "elasticmem does not support the SWA req ring"

        max_size_full = self._max_size_full()
        self.full_attn_allocator = ElasticTokenToKVPoolAllocator(
            size,
            dtype,
            device,
            kvcache.full_kv_pool,
            need_sort,
            max_size=max_size_full,
        )
        self.swa_attn_allocator = ElasticTokenToKVPoolAllocator(
            size_swa,
            dtype,
            device,
            kvcache.swa_kv_pool,
            need_sort,
        )
        # Sized for the largest full pool a resize can produce; the trailing -1
        # keeps last_loc == -1 mapping to -1 (see SWATokenToKVPoolAllocator).
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(
                    max_size_full + self.page_size, dtype=torch.int64, device=device
                ),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )
        self.clear()
        self._kvcache.register_mapping(self.full_to_swa_index_mapping)

        self.emem_orch = ElasticMempoolOrchestrator()
        self.emem_orch.register_allocator(self)
        self.emem_orch.register_allocator(self.full_attn_allocator)
        self.emem_orch.register_allocator(self.swa_attn_allocator)

    def _max_size_full(self) -> int:
        full_layers = self._kvcache.full_kv_pool.layer_num
        swa_layers = self._kvcache.swa_kv_pool.layer_num
        total_layer_tokens = self._size_full * full_layers + self._size_swa * swa_layers
        # 2x headroom over "every SWA page moved to full" (per-slot sizes can differ).
        return 2 * max(
            total_layer_tokens // max(full_layers, 1),
            total_layer_tokens // max(swa_layers, 1),
        )

    def attach(
        self,
        *,
        tree_cache: BasePrefixCache,
        max_req_len: int,
        sliding_window_size: int,
        chunked_prefill_size: Optional[int],
    ) -> None:
        """Wire eviction and evictable lookups to the prefix cache, and set
        how far each pool may shrink (never below its startup size).

        The scheduler's startup capacities and request-length limits stay the
        admission contract, so each floor keeps a max-length request
        schedulable: the full pool holds the whole request, the SWA pool one
        prefill chunk plus the window (the whole request without chunking).
        """
        self.full_attn_allocator.set_min_size(max_req_len)
        swa_min_size = (
            max_req_len
            if chunked_prefill_size is None
            else min(sliding_window_size + chunked_prefill_size, max_req_len)
        )
        self.swa_attn_allocator.set_min_size(swa_min_size)
        if not isinstance(tree_cache, UnifiedRadixCache):
            # Radix cache disabled: nothing is evictable, tails drain by frees.
            return

        self.full_attn_allocator.bind_tree_cache_fns(
            evictable_size_fn=tree_cache.full_evictable_size,
            evict_fn=lambda num_tokens: tree_cache.evict(
                EvictParams(num_tokens=num_tokens)
            ),
            evictable_indices_fn=lambda: tree_cache.component_evictable_device_indices(
                ComponentType.FULL
            ),
        )
        self.swa_attn_allocator.bind_tree_cache_fns(
            evictable_size_fn=tree_cache.swa_evictable_size,
            evict_fn=lambda num_tokens: tree_cache.evict(
                EvictParams(swa_num_tokens=num_tokens)
            ),
            evictable_indices_fn=lambda: self._full_to_swa_indices(
                tree_cache.component_evictable_device_indices(ComponentType.SWA)
            ),
        )

    def try_resize(self) -> None:
        self.emem_orch.try_resize()

    def resize(self, config) -> None:
        raise NotImplementedError("elasticmem does not support post-capture resize")

    def swa_capacity_and_available(self, *, full_capacity, swa_capacity):
        # The scheduler's startup capacities go stale after a resize; report live ones.
        return (
            (self._size_full, self.full_available_size()),
            (self._size_swa, self.swa_available_size()),
        )

    def _full_to_swa_indices(self, full_indices: torch.Tensor) -> torch.Tensor:
        swa_indices = self.full_to_swa_index_mapping[full_indices]
        return swa_indices[swa_indices > 0]

    # ---- ElasticAllocator: the composite only re-syncs sizes ----

    def can_unmap(self) -> bool:
        return False

    def can_map(self) -> bool:
        return False

    def reduce(self) -> int:
        raise NotImplementedError()

    def expand(self, expand_size: int) -> int:
        raise NotImplementedError()

    def cu_page_to_token(self, cu_page_num: int) -> int:
        raise NotImplementedError()

    def token_usage(self) -> float:
        raise NotImplementedError()

    def update_size(self) -> None:
        self._size_full = self.full_attn_allocator.size
        self._size_swa = self.swa_attn_allocator.size
        assert self._kvcache.full_kv_pool.size == self._size_full
        assert self._kvcache.swa_kv_pool.size == self._size_swa
        self._kvcache.size = self._size_full
        self._kvcache.size_swa = self._size_swa
        # Unbacked full slots must not map to a SWA slot.
        self.full_to_swa_index_mapping[self._size_full + 1 : -1] = 0
        logger.info(
            f"elasticmem: sizes updated, full={self._size_full}, swa={self._size_swa}"
        )
