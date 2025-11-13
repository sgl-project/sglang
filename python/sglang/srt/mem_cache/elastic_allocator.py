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
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator import (
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.elasticmem_orchestrator import ElasticAllocator, cu_page_size

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ElasticTokenToKVPoolAllocator(TokenToKVPoolAllocator, ElasticAllocator):
    def __init__(self, *args, **kwargs):
        self.invalid_pages = None
        super().__init__(*args, **kwargs)

    def clear(self):
        super().clear()

        if self.invalid_pages is None:
            self.invalid_pages = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
        mask = ~torch.isin(self.free_pages, self.invalid_pages)
        self.free_pages = self.free_pages[mask]

    def disable(self, need_size: int) -> int:
        self.merge_and_sort_free()
        logger.debug(f"{(need_size, len(self.free_pages))=}")
        if need_size > len(self.free_pages):
            return 0

        select_index = self.free_pages[-need_size:].tolist()
        self.free_pages = self.free_pages[:-need_size]
        unmap_num, pass_indices, proc_indices = self._kvcache.disable(select_index)

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

        self.size = self._kvcache.size

        return unmap_num

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


class ElasticSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator, ElasticAllocator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def cu_page_to_token(
        self, cu_page_num: int, layer_num: int, state_memsize: int
    ) -> int:
        cu_mem = cu_page_num * cu_page_size
        cu_mem_per_kv_layer = cu_mem // 2 // layer_num
        token_num = cu_mem_per_kv_layer // state_memsize
        return token_num

    def transfer_token(self, token_num: int, swa_to_full: bool = True) -> None:
        allocator_disable = (
            self.swa_attn_allocator if swa_to_full else self.full_attn_allocator
        )
        allocator_enable = (
            self.full_attn_allocator if swa_to_full else self.swa_attn_allocator
        )

        unmap_num = allocator_disable.disable(token_num)
        need_size = self.cu_page_to_token(
            unmap_num,
            allocator_enable._kvcache.layer_num,
            allocator_enable._kvcache.state_memsize,
        )
        map_num = allocator_enable.enable(need_size)

        self._size_swa = self.swa_attn_allocator.size
        self._size_full = self.full_attn_allocator.size

        logger.info(f"{(unmap_num, map_num)=}, {(self._size_swa, self._size_full)=}")

    def disable(self, need_size: int) -> int:
        return 0

    def enable(self, need_size: int) -> int:
        return 0
