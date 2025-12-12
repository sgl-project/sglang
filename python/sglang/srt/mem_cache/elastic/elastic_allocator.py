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


class ElasticTokenToKVPoolAllocator(TokenToKVPoolAllocator, ElasticAllocator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.need_sort = True
        logger.debug(
            f"ElasticTokenToKVPoolAllocator {self._kvcache.pool_name} initialized"
        )

    @override
    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            super().merge_and_sort_free()
        else:
            self.free_pages, _ = torch.sort(self.free_pages)

    # TODO: a more efficient way
    @override
    def alloc(self, need_size: int):
        self.merge_and_sort_free()
        return super().alloc(need_size)

    @override
    def can_unmap(self) -> bool:
        if self.token_usage() > 0.9:
            return False

        self.evict(self.evictable_size())
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
    def reduce(self) -> int:
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
                assert False

        reduce_size = tail_consecutive_size // 2
        reduce_size = reduce_size // self.page_size * self.page_size
        if reduce_size <= 0:
            return 0

        new_size = self.size - reduce_size
        self.free_pages = self.free_pages[:-reduce_size]
        unmap_num, cur_size = self._kvcache.reduce(new_size)
        logger.debug(f"{(unmap_num, cur_size)=}")
        assert cur_size == self.free_pages[-1]
        self.size = cur_size

        return unmap_num

    @override
    def expand(self, expand_size: int) -> int:
        if expand_size <= 0:
            return 0

        assert expand_size % self.page_size == 0

        new_size = self.size + expand_size
        map_num, cur_size = self._kvcache.expand(new_size)
        logger.debug(f"{(map_num, cur_size)=}")

        self.release_pages = torch.cat(
            (
                self.release_pages,
                torch.tensor(
                    range(self.size + 1, cur_size + 1),
                    dtype=torch.int64,
                    device=self.device,
                ),
            )
        )

        self.size = cur_size

        return map_num

    @override
    def cu_page_to_token(self, cu_page_num: int) -> int:
        return self._kvcache.cu_page_to_token(cu_page_num)

    @override
    def register_evict_func(self, func_evictable_size, func_evict) -> None:
        self.func_evictable_size = func_evictable_size
        self.func_evict = func_evict

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
    def update_size(self):
        # size is updated in enable()/disable()
        assert self.size == self._kvcache.size


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
    def reduce(self) -> int:
        raise NotImplementedError()

    @override
    def expand(self, expand_size: int) -> int:
        raise NotImplementedError()

    @override
    def cu_page_to_token(self, cu_page_num: int) -> int:
        raise NotImplementedError()

    @override
    def register_evict_func(self, func_evictable_size, func_evict) -> None:
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
    def update_size(self):
        self._kvcache.size = self.full_attn_allocator.size
        self._kvcache.size_swa = self.swa_attn_allocator.size
        self._size_swa = self.swa_attn_allocator.size
        self._size_full = self.full_attn_allocator.size
        self.scheduler.swa_tokens_per_layer = self._size_swa
        self.scheduler.full_tokens_per_layer = self._size_full
        logger.info(
            "ElasticSWATokenToKVPoolAllocator update_size: "
            f"{(self._size_swa, self._size_full)=}, "
            f"{(self.scheduler.swa_tokens_per_layer, self.scheduler.full_tokens_per_layer)=}"
        )


##################################################


class ElasticMambaPoolAllocator(ElasticAllocator):
    def __init__(self, mamba_pool):
        self._kvcache = mamba_pool
        logger.debug(f"ElasticMambaPoolAllocator register_allocator")

    @override  # ElasticAllocator
    def can_unmap(self) -> bool:
        if self.token_usage() > 0.8:
            return False

        self.evict(self.evictable_size())

        self._kvcache.sort_free()
        cu_page_token = (cu_page_size // self._kvcache.state_memsize + 1) * 2
        return (
            len(self._kvcache.free_slots) >= cu_page_token
            and self._kvcache.free_slots[-cu_page_token]
            == self._kvcache.size - cu_page_token
        )

    @override  # ElasticAllocator
    def can_map(self) -> bool:
        return self.token_usage() > 0.9

    @override  # ElasticAllocator
    def reduce(self) -> int:
        self._kvcache.sort_free()

        low, mid, high = 0, 0, len(self._kvcache.free_slots)
        tail_consecutive_size = -1
        while low < high:
            mid = (low + high) // 2
            tail_consecutive_value = len(self._kvcache.free_slots) - mid
            if (
                self._kvcache.free_slots[mid]
                == self._kvcache.size - tail_consecutive_value
            ):
                tail_consecutive_size = len(self._kvcache.free_slots) - mid
                high = mid
            elif (
                self._kvcache.free_slots[mid]
                < self._kvcache.size - tail_consecutive_value
            ):
                low = mid + 1
            else:
                assert False

        reduce_size = tail_consecutive_size // 2
        reduce_size = reduce_size
        if reduce_size <= 0:
            return 0

        new_size = self._kvcache.size - reduce_size
        self._kvcache.free_slots = self._kvcache.free_slots[:-reduce_size]
        unmap_num, cur_size = self._kvcache.reduce(new_size)
        logger.debug(f"{(unmap_num, cur_size)=}")
        assert cur_size == self._kvcache.free_slots[-1] + 1

        return unmap_num

    @override  # ElasticAllocator
    def expand(self, expand_size: int) -> int:
        if expand_size <= 0:
            return 0

        old_size = self._kvcache.size
        new_size = self._kvcache.size + expand_size
        map_num, cur_size = self._kvcache.expand(new_size)
        logger.debug(f"{(map_num, cur_size)=}")
        self._kvcache.free_slots = torch.cat(
            (
                self._kvcache.free_slots,
                torch.tensor(
                    range(old_size, cur_size),
                    dtype=self._kvcache.free_slots.dtype,
                    device=self._kvcache.free_slots.device,
                ),
            )
        )

        return map_num

    @override  # ElasticAllocator
    def cu_page_to_token(self, cu_page_num: int) -> int:
        return self._kvcache.cu_page_to_token(cu_page_num)

    @override  # ElasticAllocator
    def register_evict_func(self, func_evictable_size, func_evict) -> None:
        self.func_evictable_size = func_evictable_size
        self.func_evict = func_evict

    @override  # ElasticAllocator
    def token_usage(self) -> float:
        num_used = self._kvcache.size - (
            len(self._kvcache.free_slots) + self.evictable_size()
        )
        return num_used / self._kvcache.size

    @override  # ElasticAllocator
    def evictable_size(self) -> int:
        return self.func_evictable_size()

    @override  # ElasticAllocator
    def evict(self, evictable_size: int) -> None:
        self.func_evict(evictable_size)

    @override  # ElasticAllocator
    def update_size(self) -> None:
        pass


class ElasticHybridLinearKVPoolAllocator(ElasticAllocator):
    def __init__(self, emem_orch, token_to_kv_pool_allocator):
        self.emem_orch = emem_orch
        self.full_allocator = token_to_kv_pool_allocator
        self.mamba_allocator = ElasticMambaPoolAllocator(
            self.full_allocator._kvcache.mamba_pool
        )
        logger.debug(f"ElasticHybridLinearKVPoolAllocator initialized")

        self.emem_orch.register_allocator(self)
        self.emem_orch.register_allocator(self.full_allocator)
        self.emem_orch.register_allocator(self.mamba_allocator)
        logger.debug(f"ElasticHybridLinearKVPoolAllocator register_allocator")

    @override
    def register_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler
        self.full_allocator.register_evict_func(
            func_evictable_size=self.scheduler.tree_cache.full_evictable_size,
            func_evict=lambda evictable_size: self.scheduler.tree_cache.evict(
                evictable_size
            ),
        )
        self.mamba_allocator.register_evict_func(
            func_evictable_size=self.scheduler.tree_cache.mamba_evictable_size,
            func_evict=lambda evictable_size: self.scheduler.tree_cache.evict_mamba(
                evictable_size
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
    def register_evict_func(self, func_evictable_size, func_evict) -> None:
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
    def update_size(self):
        logger.info(
            "ElasticHybridLinearKVPoolAllocator update_size: "
            f"{(self.full_allocator.size, self.mamba_allocator._kvcache.size)=}"
        )
