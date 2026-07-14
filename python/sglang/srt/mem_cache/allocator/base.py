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

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    supports_page_aligned_alloc: bool = False
    supports_spec_page_aligned_alloc: bool = False

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
        self.is_not_in_free_group = True
        self.free_group = []

    @property
    def size_full(self):
        return self.size

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def _validate_free_index_metadata(
        self,
        free_index: torch.Tensor,
        *,
        page_size: int,
    ) -> None:
        assert isinstance(free_index, torch.Tensor), (
            f"{type(self).__name__}.free expects a torch.Tensor, "
            f"got {type(free_index).__name__}"
        )
        assert free_index.ndim == 1, (
            f"{type(self).__name__}.free expects a 1-D tensor, "
            f"got shape={free_index.shape}"
        )
        assert free_index.dtype in (torch.int32, torch.int64), (
            f"{type(self).__name__}.free expects int32 or int64 indices, "
            f"got dtype={free_index.dtype}"
        )
        expected_device = torch.device(self.device)
        actual_device = free_index.device
        device_matches = actual_device == expected_device or (
            expected_device.index is None
            and actual_device.type == expected_device.type
        )
        assert device_matches, (
            f"{type(self).__name__}.free indices are on {actual_device}, "
            f"expected {expected_device}"
        )
        assert isinstance(page_size, int) and page_size > 0, (
            f"{type(self).__name__}.free requires a positive integer page size, "
            f"got {page_size}"
        )
        assert free_index.numel() % page_size == 0, (
            f"{type(self).__name__}.free requires complete page blocks: "
            f"numel={free_index.numel()}, page_size={page_size}"
        )

    def _extract_validated_free_page_ids(
        self,
        free_index: torch.Tensor,
        *,
        page_size: int,
    ) -> torch.Tensor:
        return free_index[::page_size].to(dtype=torch.int64) // page_size

    def _debug_validate_free_page_blocks(
        self,
        free_index: torch.Tensor,
        page_ids: torch.Tensor,
        *,
        page_size: int,
        owner_page_start: int,
        owner_page_end: int,
        live_page_table: torch.Tensor | None = None,
    ) -> None:
        if free_index.numel() == 0 or not envs.SGLANG_DEBUG_MEMORY_POOL.get():
            return

        assert 0 <= owner_page_start < owner_page_end
        blocks = free_index.reshape(-1, page_size).to(dtype=torch.int64)
        representatives = blocks[:, 0]
        expected_blocks = representatives[:, None] + torch.arange(
            page_size,
            dtype=torch.int64,
            device=free_index.device,
        )
        sorted_page_ids = torch.sort(page_ids).values
        valid_contract = (
            torch.all(representatives % page_size == 0)
            & torch.all(blocks == expected_blocks)
            & torch.all(page_ids >= owner_page_start)
            & torch.all(page_ids < owner_page_end)
            & torch.all(sorted_page_ids[1:] != sorted_page_ids[:-1])
        )
        torch._assert_async(
            valid_contract,
            f"{type(self).__name__}.free requires unique complete page blocks "
            f"in owner domain [{owner_page_start}, {owner_page_end})",
        )

        if live_page_table is not None:
            safe_page_ids = page_ids.clamp(
                min=0,
                max=live_page_table.shape[0] - 1,
            )
            valid_live_binding = (
                (page_ids >= owner_page_start)
                & (page_ids < owner_page_end)
                & (live_page_table[safe_page_ids] >= 0)
            )
            torch._assert_async(
                torch.all(valid_live_binding),
                f"{type(self).__name__}.free requires live page bindings "
                f"in owner domain [{owner_page_start}, {owner_page_end})",
            )

    def _debug_validate_free_index(
        self,
        free_index: torch.Tensor,
        *,
        page_size: int,
        owner_page_start: int,
        owner_page_end: int,
        live_page_table: torch.Tensor | None = None,
    ) -> None:
        if free_index.numel() == 0 or not envs.SGLANG_DEBUG_MEMORY_POOL.get():
            return

        free_page_ids = self._extract_validated_free_page_ids(
            free_index,
            page_size=page_size,
        )
        self._debug_validate_free_page_blocks(
            free_index,
            free_page_ids,
            page_size=page_size,
            owner_page_start=owner_page_start,
            owner_page_end=owner_page_end,
            live_page_table=live_page_table,
        )

    def _get_free_page_owner_bounds(self) -> tuple[int, int]:
        raise NotImplementedError(
            f"{type(self).__name__} does not define a free-page owner domain"
        )

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

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
