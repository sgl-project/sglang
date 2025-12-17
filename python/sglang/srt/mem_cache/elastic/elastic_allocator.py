from __future__ import annotations

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

import logging
import time
from typing import TYPE_CHECKING, Set, override

import torch

from sglang.srt.mem_cache.allocator import (
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import (
    ElasticAllocator,
    can_map_threshold,
    can_unmap_threshold,
    cu_page_size,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


from torch.utils.cpp_extension import load_inline

source_get_tail_consecutive_start = """
int64_t get_tail_consecutive_start(at::Tensor unused_pages) {
    int64_t n = unused_pages.numel();
    if (n == 0) return 0;

    auto acc = unused_pages.accessor<bool, 1>();
    int64_t i;
    for (i = n - 1; i >= 0 && acc[i]; i--);

    return i + 1;
}
"""

elasticmem_utils = load_inline(
    name="elasticmem_utils",
    cpp_sources=[source_get_tail_consecutive_start],
    functions=["get_tail_consecutive_start"],
)


class ElasticTokenToKVPoolAllocator(TokenToKVPoolAllocator, ElasticAllocator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cu_page_token_num = cu_page_size // self._kvcache.state_memsize + 1
        logger.debug(
            f"ElasticTokenToKVPoolAllocator {self._kvcache.pool_name} initialized, "
            f"{self.cu_page_token_num=}"
        )

    @override
    def clear(self):
        super().clear()
        self.unused_pages = torch.ones((self.size + 1,), dtype=torch.bool)

    @override
    def alloc(self, need_size: int):
        self_unused_pages = self.unused_pages[1:]
        self_unused_pages = self_unused_pages[self_unused_pages == True]
        assert (self.available_size() + self.evictable_size()) == len(
            self_unused_pages
        ), (
            f"{self.available_size()=} + {self.evictable_size()=} = {self.available_size() + self.evictable_size()}, "
            f"{len(self_unused_pages)=}\n"
            f"{need_size=}\n"
        )
        select_index = super().alloc(need_size)
        self.unused_pages[select_index.cpu()] = False
        self_unused_pages = self.unused_pages[1:]
        self_unused_pages = self_unused_pages[self_unused_pages == True]
        assert (self.available_size() + self.evictable_size()) == len(
            self_unused_pages
        ), (
            f"{self.available_size()=} + {self.evictable_size()=} = {self.available_size() + self.evictable_size()}, "
            f"{len(self_unused_pages)=}\n"
            f"{need_size=}, {select_index=}\n"
        )
        return select_index

    @override
    def free(self, free_index: torch.Tensor):
        super().free(free_index)
        if self.is_not_in_free_group:
            self.unused_pages[free_index.cpu()] = True
        self_unused_pages = self.unused_pages[1:]
        self_unused_pages = self_unused_pages[self_unused_pages == True]
        # Free first, then delete_leaf updates evictable_size—may temporarily exceed len(self_unused_pages).
        assert (self.available_size() + self.evictable_size()) >= len(
            self_unused_pages
        ), (
            f"{self.is_not_in_free_group=}, "
            f"{self.available_size()=} + {self.evictable_size()=} = {self.available_size() + self.evictable_size()}, "
            f"{len(self_unused_pages)=}"
        )

    def add_unused(self, indices: torch.Tensor):
        self.unused_pages[indices.cpu()] = True
        self_unused_pages = self.unused_pages[1:]
        self_unused_pages = self_unused_pages[self_unused_pages == True]
        assert (self.available_size() + self.evictable_size()) == len(
            self_unused_pages
        ), (
            f"{self.available_size()=} + {self.evictable_size()=} = {self.available_size() + self.evictable_size()}, "
            f"{len(self_unused_pages)=}"
        )

    def rm_unused(self, indices: torch.Tensor):
        self.unused_pages[indices.cpu()] = False
        self_unused_pages = self.unused_pages[1:]
        self_unused_pages = self_unused_pages[self_unused_pages == True]
        assert (self.available_size() + self.evictable_size()) == len(
            self_unused_pages
        ), (
            f"{self.available_size()=} + {self.evictable_size()=} = {self.available_size() + self.evictable_size()}, "
            f"{len(self_unused_pages)=}"
        )

    @override
    def can_unmap(self) -> bool:
        if self.token_usage() > can_unmap_threshold:
            return False

        tail_consecutive_start = elasticmem_utils.get_tail_consecutive_start(
            self.unused_pages
        )
        tail_consecutive_tokens = self.size + 1 - tail_consecutive_start
        return tail_consecutive_tokens > 2 * self.cu_page_token_num

    @override
    def can_map(self) -> bool:
        return self.token_usage() > can_map_threshold

    @override
    def reduce(self) -> int:
        start_time = time.perf_counter()

        tail_consecutive_start = elasticmem_utils.get_tail_consecutive_start(
            self.unused_pages
        )
        tail_consecutive_tokens = self.size + 1 - tail_consecutive_start
        reduce_size = tail_consecutive_tokens // 2
        reduce_size = reduce_size // self.page_size * self.page_size
        if reduce_size <= 0:
            return 0

        # Evict pages with indices > reduce_start
        reduce_start = self.size + 1 - reduce_size
        free_indices = self.free_pages[self.free_pages >= reduce_start].tolist()
        evict_indices = set(range(reduce_start, self.size + 1))
        evict_indices.difference_update(free_indices)
        self.evict_indices(evict_indices)
        assert (
            len(self.free_pages[self.free_pages >= reduce_start])
            == self.size + 1 - reduce_start
        ), (
            f"{(len(self.free_pages[self.free_pages >= reduce_start]))=}, "
            f"{(self.size + 1)=} - {reduce_start=} = {self.size + 1 - reduce_start}"
        )

        new_size = self.size - reduce_size
        self.free_pages = self.free_pages[self.free_pages < reduce_start]
        self.unused_pages = self.unused_pages[: new_size + 1]
        unmap_num, cur_size = self._kvcache.reduce(new_size)
        logger.debug(f"{reduce_size=}, {(unmap_num, cur_size)=}")
        self.size = cur_size

        logger.info(f"reduce took {(time.perf_counter() - start_time) * 1000} ms")
        return unmap_num

    @override
    def expand(self, expand_size: int) -> int:
        start_time = time.perf_counter()

        if expand_size <= 0:
            return 0

        assert expand_size % self.page_size == 0

        new_size = self.size + expand_size
        map_num, cur_size = self._kvcache.expand(new_size)
        logger.debug(f"{expand_size=}, {(map_num, cur_size)=}")

        self.free_pages = torch.cat(
            (
                self.free_pages,
                torch.tensor(
                    range(self.size + 1, cur_size + 1),
                    dtype=torch.int64,
                    device=self.device,
                ),
            )
        )
        self.unused_pages = torch.cat(
            (
                self.unused_pages,
                torch.ones((cur_size - self.size,), dtype=torch.bool),
            )
        )

        self.size = cur_size

        logger.info(f"expand took {(time.perf_counter() - start_time) * 1000} ms")
        return map_num

    @override
    def cu_page_to_token(self, cu_page_num: int) -> int:
        return self._kvcache.cu_page_to_token(cu_page_num)

    @override
    def register_evict_func(
        self, func_evictable_size, func_evict, func_evict_indices
    ) -> None:
        self.func_evictable_size = func_evictable_size
        self.func_evict = func_evict
        self.func_evict_indices = func_evict_indices

    @override
    def token_usage(self) -> float:
        num_used = self.size - (self.available_size() + self.evictable_size())
        return num_used / self.size

    @override
    def evictable_size(self) -> int:
        return self.func_evictable_size()

    @override
    def evict(self, evictable_size: int) -> None:
        self.func_evict(evictable_size)

    @override
    def evict_indices(self, indices: Set[int]) -> None:
        self.func_evict_indices(indices)

    @override
    def update_size(self):
        # size is updated in enable()/disable()
        assert self.size == self._kvcache.size

    @override
    def defragmentation(self) -> bool:
        mid_index = self.size // 2

        if len(self.free_pages) > 0 and self.free_pages[0] < mid_index:
            return False

        self.evict(self.evictable_size())
        self.free_pages, _ = torch.sort(self.free_pages)
        return True


class ElasticSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator, ElasticAllocator):
    def __init__(self, *args, emem_orch, **kwargs):
        self.emem_orch = emem_orch

        super().__init__(*args, **kwargs)
        logger.debug(f"ElasticSWATokenToKVPoolAllocator initialized")

        self.emem_orch.register_allocator(self)
        self.emem_orch.register_allocator(self.full_attn_allocator)
        self.emem_orch.register_allocator(self.swa_attn_allocator)
        logger.debug(f"ElasticSWATokenToKVPoolAllocator register_allocator")

    def _create_allocator(self):
        self.full_attn_allocator = ElasticTokenToKVPoolAllocator(
            self._size_full,
            self.dtype,
            self.device,
            self._kvcache.full_kv_pool,
            self.need_sort,
        )
        self.swa_attn_allocator = ElasticTokenToKVPoolAllocator(
            self._size_swa,
            self.dtype,
            self.device,
            self._kvcache.swa_kv_pool,
            self.need_sort,
        )
        # oversubscribe as full pool may expand
        total_tokens = (
            self._size_full * self._kvcache.full_kv_pool.layer_num
            + self._size_swa * self._kvcache.swa_kv_pool.layer_num
        )
        oversubscribe_tokens = (
            max(
                total_tokens // self._kvcache.full_kv_pool.layer_num,
                total_tokens // self._kvcache.swa_kv_pool.layer_num,
            )
            * 2
        )
        self.full_to_swa_index_mapping = torch.empty(
            oversubscribe_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        logger.debug(
            f"{(self.full_to_swa_index_mapping.numel() * self.full_to_swa_index_mapping.dtype.itemsize // (1<<20))=}"
        )

    def add_unused(self, indices: torch.Tensor):
        self.full_attn_allocator.add_unused(indices)

    def rm_unused(self, indices: torch.Tensor):
        self.full_attn_allocator.rm_unused(indices)

    def full_indices_to_swa_indices(self, full_index: torch.Tensor):
        swa_indices = self.full_to_swa_index_mapping[full_index]
        swa_indices = swa_indices[swa_indices > 0]
        return swa_indices

    def add_unused_swa(self, indices: torch.Tensor):
        self.swa_attn_allocator.add_unused(self.full_indices_to_swa_indices(indices))

    def rm_unused_swa(self, indices: torch.Tensor):
        self.swa_attn_allocator.rm_unused(self.full_indices_to_swa_indices(indices))

    @override
    def register_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler
        self.full_attn_allocator.register_evict_func(
            func_evictable_size=self.scheduler.tree_cache.full_evictable_size,
            func_evict=lambda evictable_size: self.scheduler.tree_cache.evict(
                evictable_size, 0
            ),
            func_evict_indices=lambda indices: self.scheduler.tree_cache.evict_indices(
                indices, set()
            ),
        )
        self.swa_attn_allocator.register_evict_func(
            func_evictable_size=self.scheduler.tree_cache.swa_evictable_size,
            func_evict=lambda evictable_size: self.scheduler.tree_cache.evict(
                0, evictable_size
            ),
            func_evict_indices=lambda indices: self.scheduler.tree_cache.evict_indices(
                set(), indices
            ),
        )

    @override
    def can_unmap(self) -> bool:
        return False

    @override
    def can_map(self) -> bool:
        return False

    @override
    def reduce(self) -> int:
        raise NotImplementedError()

    @override
    def expand(self, expand_size: int) -> int:
        raise NotImplementedError()

    @override
    def cu_page_to_token(self, cu_page_num: int) -> int:
        raise NotImplementedError()

    @override
    def register_evict_func(
        self, func_evictable_size, func_evict, func_evict_indices
    ) -> None:
        raise NotImplementedError()

    @override
    def token_usage(self) -> float:
        return 1

    @override
    def evictable_size(self) -> int:
        raise NotImplementedError()

    @override
    def evict(self, evictable_size: int) -> None:
        raise NotImplementedError()

    @override
    def evict_indices(self, indices: Set[int]) -> None:
        raise NotImplementedError()

    @override
    def update_size(self):
        self._kvcache.size == self.full_attn_allocator.size
        self._kvcache.size_swa == self.swa_attn_allocator.size
        self._size_swa = self.swa_attn_allocator.size
        assert all(
            self.full_to_swa_index_mapping[
                self._size_full + 1 : self.full_attn_allocator.size + 1
            ]
            == 0
        )
        self._size_full = self.full_attn_allocator.size
        self.scheduler.swa_tokens_per_layer = self._size_swa
        self.scheduler.full_tokens_per_layer = self._size_full
        self.full_to_swa_index_mapping[self._size_full + 1 :] = 0
        logger.info(
            "ElasticSWATokenToKVPoolAllocator update_size: "
            f"{(self._size_swa, self._size_full)=}, "
            f"{(self.scheduler.swa_tokens_per_layer, self.scheduler.full_tokens_per_layer)=}"
        )

    @override
    def defragmentation(self) -> bool:
        raise NotImplementedError()
