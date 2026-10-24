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

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self._stage_releases = torch.device(device).type == "cpu"
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.free_group = None
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.staged_pages: list[torch.Tensor] = []
        self.num_staged_pages = 0

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages) + self.num_staged_pages

    def get_all_free_pages(self):
        if not self._stage_releases:
            return super().get_all_free_pages()
        if not self.staged_pages:
            return self.free_pages
        return torch.cat((self.free_pages, *self.staged_pages))

    def merge_and_sort_free(self):
        if not self._stage_releases:
            return super().merge_and_sort_free()
        if not self.staged_pages:
            return
        self.free_pages = self.get_all_free_pages()
        if self.need_sort:
            self.free_pages, _ = torch.sort(self.free_pages)
        self.staged_pages = []
        self.num_staged_pages = 0

    def alloc(self, need_size: int):
        if self._stage_releases and need_size > len(self.free_pages):
            if need_size > self.available_size():
                return None
            self.merge_and_sort_free()
        elif self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.free_group is None:
            if self._stage_releases:
                # CPU copies scale with the growing free list on every release.
                # Own the released view; callers can mutate request-token rows.
                self.staged_pages.append(free_index.clone())
                self.num_staged_pages += free_index.numel()
            elif self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(self._copy_for_free_group(free_index))

    def free_page_ids(self, page_ids: torch.Tensor):
        # page_size == 1: page ids are token ids.
        self.free(page_ids)

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        return self._kvcache.get_cpu_copy(
            indices,
            mamba_indices=mamba_indices,
            req_pool_index=req_pool_index,
        )

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu,
            indices,
            mamba_indices=mamba_indices,
            req_pool_index=req_pool_index,
        )
