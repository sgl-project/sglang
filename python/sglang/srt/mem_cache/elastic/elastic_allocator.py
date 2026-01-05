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
from typing import TYPE_CHECKING, override

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
        self.candidate_size = self.size
        self.candidate_unmap_pages = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )

    def available_size(self):
        return (
            len(self.free_pages)
            + len(self.release_pages)
            + len(self.candidate_unmap_pages)
        )

    def _check_unused_pages(self):
        return
        self_unused_pages = self.unused_pages[1:]
        self_unused_pages = self_unused_pages[self_unused_pages == True]
        assert (self.available_size() + self.evictable_size()) == len(
            self_unused_pages
        ), (
            f"{len(self.free_pages)=}, {len(self.release_pages)=}, {len(self.candidate_unmap_pages)=}, "
            f"{self.available_size()=} + {self.evictable_size()=} = {self.available_size() + self.evictable_size()}; "
            f"{len(self_unused_pages)=}, {len(self.unused_pages)=}"
        )

    def _unmap_candidate_pre_alloc(self, need_size: int):
        if self.candidate_size == self.size:
            return

        available_size = len(self.free_pages) + len(self.release_pages)
        if need_size <= available_size:
            return

        self._evict_tail()
        evict_size = min(self.evictable_size(), need_size)
        self.evict(evict_size)

        available_size = len(self.free_pages) + len(self.release_pages)
        if need_size <= available_size:
            return

        extra_need_size = need_size - available_size
        logger.debug(
            f"{extra_need_size=} {available_size=} {self.evictable_size()=} {len(self.candidate_unmap_pages)=}"
        )

        if self.need_sort:
            self.release_pages = torch.cat(
                (
                    self.release_pages,
                    self.candidate_unmap_pages[:extra_need_size],
                )
            )
        else:
            self.free_pages = torch.cat(
                (
                    self.free_pages,
                    self.candidate_unmap_pages[:extra_need_size],
                )
            )

        self.candidate_unmap_pages = self.candidate_unmap_pages[extra_need_size:]

        # DEBUG
        self_candidate_unmap_pages = self.candidate_unmap_pages.tolist()
        # assert len(set(self_candidate_unmap_pages)) == len(self_candidate_unmap_pages)

    @override
    def alloc(self, need_size: int):
        self._check_unused_pages()
        self._unmap_candidate_pre_alloc(need_size)
        self._check_unused_pages()

        select_index = super().alloc(need_size)
        self.unused_pages[select_index.cpu()] = False

        self._check_unused_pages()

        return select_index

    def _unmap_candidate_post_free(self, free_index: torch.Tensor):
        if self.candidate_size == self.size:
            return

        if not (free_index > self.candidate_size).any():
            return

        if self.need_sort:
            unmap_candidate_mask = self.release_pages > self.candidate_size
            self.candidate_unmap_pages = torch.cat(
                (self.candidate_unmap_pages, self.release_pages[unmap_candidate_mask])
            )
            self.release_pages = self.release_pages[~unmap_candidate_mask]
        else:
            unmap_candidate_mask = self.free_pages > self.candidate_size
            self.candidate_unmap_pages = torch.cat(
                (self.candidate_unmap_pages, self.free_pages[unmap_candidate_mask])
            )
            self.free_pages = self.free_pages[~unmap_candidate_mask]

        # DEBUG
        self_free_pages = self.free_pages.tolist()
        self_candidate_unmap_pages = self.candidate_unmap_pages.tolist()
        # assert len(set(self_free_pages)) == len(self_free_pages)
        # assert len(set(self_candidate_unmap_pages)) == len(self_candidate_unmap_pages)
        # assert len(set(self_free_pages) & set(self_candidate_unmap_pages)) == 0

    @override
    def free(self, free_index: torch.Tensor):
        super().free(free_index)

        if not self.is_not_in_free_group:
            return

        # Free first, then delete_leaf updates evictable_size - may temporarily exceed len(self_unused_pages), skip self._check_unused_pages().
        self.unused_pages[free_index.cpu()] = True

        self._unmap_candidate_post_free(free_index)

    def add_unused(self, indices: torch.Tensor):
        self.unused_pages[indices.cpu()] = True
        self._check_unused_pages()

    def rm_unused(self, indices: torch.Tensor):
        self.unused_pages[indices.cpu()] = False
        self._check_unused_pages()

    @override
    def can_be_candidate(self) -> bool:
        return True

    def _evict_tail(self):
        start_time = time.perf_counter()
        # assert all(self.free_pages <= self.candidate_size)
        # assert all(self.release_pages <= self.candidate_size)
        evict_indices = self.unused_pages.nonzero(as_tuple=True)[0]
        evict_indices = evict_indices[evict_indices > self.candidate_size]

        if self.candidate_unmap_pages.numel() == evict_indices.numel():
            return

        if self.candidate_unmap_pages.numel() > 0:
            evict_indices = evict_indices[
                ~torch.isin(evict_indices, self.candidate_unmap_pages.cpu())
            ]
        logger.debug(
            f"emem evict_tail prepare took {(time.perf_counter() - start_time) * 1000} ms"
        )

        while (self.evictable_size() >= len(evict_indices)) and (
            len(evict_indices) > 0
        ):
            self.evict(len(evict_indices))
            evict_indices = evict_indices[
                ~torch.isin(evict_indices, self.candidate_unmap_pages.cpu())
            ]

        logger.info(
            f"emem evict_tail took {(time.perf_counter() - start_time) * 1000} ms"
        )

    @override
    def mark_unmap_candidate(self, is_candidate: bool) -> ElasticAllocator:
        # is_candidate == False
        if not is_candidate:
            assert self.candidate_size <= self.size
            if self.candidate_size < self.size:
                self.candidate_size = self.size
                if self.need_sort:
                    self.release_pages = torch.cat(
                        (self.release_pages, self.candidate_unmap_pages)
                    )
                else:
                    self.free_pages = torch.cat(
                        (self.free_pages, self.candidate_unmap_pages)
                    )

                self.candidate_unmap_pages = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
            assert len(self.candidate_unmap_pages) == 0
            logger.info(
                f"mark_unmap_candidate: {self._kvcache.pool_name=}, {is_candidate=}, {self.size=}, {self.candidate_size=}"
            )
            return None

        # is_candidate == True
        assert (
            self.candidate_size == self.size and len(self.candidate_unmap_pages) == 0
        ), f"{self.candidate_size=}, {self.size=}, {len(self.candidate_unmap_pages)=}"
        token_usage = self.token_usage()
        new_size = int((token_usage + 1) / 2 * self.size)

        if self.size - new_size < 2 * self.cu_page_token_num:
            logger.info(
                f"mark_unmap_candidate: {self._kvcache.pool_name=}, {is_candidate=}, {self.size=}, {self.candidate_size=}"
            )
            return None

        self.candidate_size = new_size

        unmap_candidate_mask = self.release_pages > self.candidate_size
        self.candidate_unmap_pages = torch.cat(
            (self.candidate_unmap_pages, self.release_pages[unmap_candidate_mask])
        )
        self.release_pages = self.release_pages[~unmap_candidate_mask]
        unmap_candidate_mask = self.free_pages > self.candidate_size
        self.candidate_unmap_pages = torch.cat(
            (self.candidate_unmap_pages, self.free_pages[unmap_candidate_mask])
        )
        self.free_pages = self.free_pages[~unmap_candidate_mask]

        logger.info(
            f"mark_unmap_candidate: {self._kvcache.pool_name=}, {self.size=}, {self.candidate_size=}"
        )
        return self

    @override
    def can_unmap(self) -> bool:
        return self.token_usage() < can_unmap_threshold

    @override
    def can_do_unmap(self) -> bool:
        tail_consecutive_start = elasticmem_utils.get_tail_consecutive_start(
            self.unused_pages
        )
        return tail_consecutive_start <= self.candidate_size + 1

    @override
    def can_map(self) -> bool:
        return self.token_usage() > can_map_threshold

    @override
    def reduce(self) -> int:
        start_time = time.perf_counter()

        self._evict_tail()

        # DEBUG
        self_candidate_unmap_pages = self.candidate_unmap_pages.tolist()
        # assert len(set(self_candidate_unmap_pages)) == len(
        #     self_candidate_unmap_pages
        # ), f"{(len(self.candidate_unmap_pages), min(self.candidate_unmap_pages), max(self.candidate_unmap_pages), self.size, self.candidate_size)=}"

        assert (
            len(self.candidate_unmap_pages) == self.size - self.candidate_size
        ), f"{(len(self.candidate_unmap_pages), min(self.candidate_unmap_pages), max(self.candidate_unmap_pages), self.size, self.candidate_size)=}"

        new_size = self.candidate_size
        self.candidate_unmap_pages = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        self.unused_pages = self.unused_pages[: new_size + 1]
        unmap_num, cur_size = self._kvcache.reduce(self.candidate_size)
        logger.debug(f"{(self.size, self.candidate_size, unmap_num, cur_size)=}")
        self.size = cur_size

        logger.info(f"reduce took {(time.perf_counter() - start_time) * 1000} ms")
        return unmap_num

    @override
    def expand(self, expand_size: int) -> int:
        start_time = time.perf_counter()

        assert self.candidate_size == self.size and len(self.candidate_unmap_pages) == 0

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
        self.candidate_size = self.size

        logger.info(f"expand took {(time.perf_counter() - start_time) * 1000} ms")
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
    def __init__(self, *args, emem_orch, sliding_window_size, **kwargs):
        self.emem_orch = emem_orch
        self.sliding_window_size = sliding_window_size

        super().__init__(*args, **kwargs)
        logger.debug(
            f"ElasticSWATokenToKVPoolAllocator initialized, {self.sliding_window_size=}"
        )

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
        self._kvcache.size == self.full_attn_allocator.size
        self._kvcache.size_swa == self.swa_attn_allocator.size
        self._size_swa = self.swa_attn_allocator.size
        # assert all(
        #     self.full_to_swa_index_mapping[
        #         self._size_full + 1 : self.full_attn_allocator.size + 1
        #     ]
        #     == 0
        # )
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
    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        swa_indices = self.full_to_swa_index_mapping[kv_indices].to(torch.int32)
        swa_indices[: -self.sliding_window_size] = 0
        return swa_indices
