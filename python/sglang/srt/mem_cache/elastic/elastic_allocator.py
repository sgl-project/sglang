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
from typing import TYPE_CHECKING, override

import torch
from torch.utils.cpp_extension import load_inline

from sglang.srt.mem_cache.allocator import (
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import (
    ElasticAllocator,
    cu_page_size,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

source_get_tail_consecutive = """
int64_t get_tail_consecutive(at::Tensor free_pages, int64_t cu_page_token, int64_t size) {
    py::gil_scoped_release release;
    int64_t tail_consecutive_size = cu_page_token;
    int64_t total_pages = free_pages.numel();
    while (
        tail_consecutive_size <= total_pages &&
        free_pages[total_pages - tail_consecutive_size].item<int64_t>() == size - tail_consecutive_size + 1
    ) {
        tail_consecutive_size += cu_page_token;
    }
    tail_consecutive_size -= cu_page_token;
    return tail_consecutive_size;
}
"""
elasticmem_utils = load_inline(
    name="elasticmem_extension",
    cpp_sources=[source_get_tail_consecutive],
    functions=["get_tail_consecutive"],
)


class ElasticTokenToKVPoolAllocator(TokenToKVPoolAllocator, ElasticAllocator):
    def __init__(self, *args, **kwargs):
        self.invalid_pages = None
        super().__init__(*args, **kwargs)
        self.need_sort = True
        logger.debug(
            f"ElasticTokenToKVPoolAllocator {self._kvcache.pool_name} initialized"
        )

    def clear(self):
        super().clear()

        if self.invalid_pages is None:
            self.invalid_pages = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
        mask = ~torch.isin(self.free_pages, self.invalid_pages)
        self.free_pages = self.free_pages[mask]

    def token_usage(self) -> float:
        num_used = self.size - (self.available_size() + self.evictable_size())
        return num_used / self.size

    @override
    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            super().merge_and_sort_free()
        else:
            self.free_pages, _ = torch.sort(self.free_pages)

    @override
    def can_unmap(self) -> bool:
        if self.token_usage() < 0.9:
            return True

        self.merge_and_sort_free()
        cu_page_token = (cu_page_size // self._kvcache.state_memsize + 1) * 2
        return (
            len(self.free_pages) >= cu_page_token
            and self.free_pages[-cu_page_token] == self.size - cu_page_token + 1
        )

    @override
    def can_map(self) -> bool:
        return self.token_usage() > 0.9

    @override
    def disable(self) -> int:
        self.merge_and_sort_free()
        low, mid, high = 0, 0, len(self.free_pages)
        tail_consecutive_size = -1
        while low < high:
            mid = (low + high) // 2
            if self.free_pages[mid] == self.size - (len(self.free_pages) - mid) + 1:
                tail_consecutive_size = len(self.free_pages) - mid
                high = mid
            elif self.free_pages[mid] < self.size - (len(self.free_pages) - mid) + 1:
                low = mid + 1
            else:
                assert (
                    False
                ), f"{self.size=}, {mid=}, {len(self.free_pages)=}, {self.free_pages[mid:]}"

        need_size = tail_consecutive_size // 2
        if need_size <= 0:
            return 0

        select_index = self.free_pages[-need_size:].tolist()
        self.free_pages = self.free_pages[:-need_size]
        unmap_num, pass_indices, proc_indices = self._kvcache.disable(select_index)
        logger.debug(f"{(unmap_num, len(pass_indices), len(proc_indices))=}")

        self.free_pages = torch.cat(
            (
                self.free_pages,
                torch.tensor(pass_indices, dtype=torch.int64, device=self.device),
            )
        )
        self.invalid_pages = torch.cat(
            (
                self.invalid_pages,
                torch.tensor(proc_indices, dtype=torch.int64, device=self.device),
            )
        )
        logger.debug(f"{len(self.invalid_pages)=}")

        self.size = self._kvcache.size

        return unmap_num

    @override
    def enable(self, need_size: int) -> int:
        expand_size = 0
        invalid_size = len(self.invalid_pages)
        if need_size > invalid_size:
            expand_size = need_size - invalid_size
        logger.debug(f"{(need_size, invalid_size, expand_size)=}")

        self.invalid_pages, _ = torch.sort(self.invalid_pages)

        select_index = self.invalid_pages[:need_size].tolist()
        select_index.extend(
            range(self.size + invalid_size, self.size + invalid_size + expand_size)
        )

        self.invalid_pages = self.invalid_pages[need_size:]
        map_num, pass_indices, proc_indices = self._kvcache.enable(select_index)

        self.invalid_pages = torch.cat(
            (
                self.invalid_pages,
                torch.tensor(pass_indices, dtype=torch.int64, device=self.device),
            )
        )
        self.release_pages = torch.cat(
            (
                self.release_pages,
                torch.tensor(proc_indices, dtype=torch.int64, device=self.device),
            )
        )

        self.size = self._kvcache.size

        return map_num

    @override
    def cu_page_to_token(self, cu_page_num: int) -> int:
        return self._kvcache.cu_page_to_token(cu_page_num)

    def register_evict_func(self, func_evictable_size, func_evict) -> None:
        self.func_evictable_size = func_evictable_size
        self.func_evict = func_evict

    @override
    def evictable_size(self) -> int:
        return self.func_evictable_size()

    @override
    def evict(self, evictable_size: int) -> None:
        self.func_evict(evictable_size)

    @override
    def update_size(self):
        # size is updated in enable()/disable()
        return


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
        self.full_to_swa_index_mapping = torch.empty(
            (self._size_full + self._size_swa + 1) * 3,
            dtype=torch.int64,
            device=self.device,
        )

    @override
    def register_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler
        self.full_attn_allocator.register_evict_func(
            func_evictable_size=self.scheduler.tree_cache.full_evictable_size,
            func_evict=lambda evictable_size: self.scheduler.tree_cache.evict(
                evictable_size, 0
            ),
        )
        self.swa_attn_allocator.register_evict_func(
            func_evictable_size=self.scheduler.tree_cache.swa_evictable_size,
            func_evict=lambda evictable_size: self.scheduler.tree_cache.evict(
                0, evictable_size
            ),
        )

    @override
    def can_unmap(self) -> bool:
        return False

    @override
    def can_map(self) -> bool:
        return False

    @override
    def disable(self) -> int:
        raise NotImplementedError()

    @override
    def enable(self, need_size: int) -> int:
        raise NotImplementedError()

    @override
    def cu_page_to_token(self, cu_page_num: int) -> int:
        raise NotImplementedError()

    @override
    def evictable_size(self) -> int:
        raise NotImplementedError()

    @override
    def evict(self, evictable_size: int) -> None:
        raise NotImplementedError()

    @override
    def update_size(self):
        self._size_swa = self.swa_attn_allocator.size
        self._size_full = self.full_attn_allocator.size
        self.scheduler.swa_tokens_per_layer = self._size_swa
        self.scheduler.full_tokens_per_layer = self._size_full
        logger.info(
            "ElasticSWATokenToKVPoolAllocator update_size: "
            f"{(self._size_swa, self._size_full)=}, "
            f"{(self.scheduler.swa_tokens_per_layer, self.scheduler.full_tokens_per_layer)=}"
        )
