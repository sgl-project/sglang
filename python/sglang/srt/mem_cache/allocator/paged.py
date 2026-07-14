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

"""
Page-aligned memory pool.
"""


from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.triton_ops.allocator import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.utils import (
    get_bool_env_var,
    get_num_new_pages,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.

    TODO: fuse last_loc into the kernel.
    """

    supports_page_aligned_alloc: bool = True
    supports_spec_page_aligned_alloc: bool = True

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
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def free(self, free_index: torch.Tensor):
        self._validate_free_index_metadata(free_index, page_size=self.page_size)
        free_page_indices = self._extract_validated_free_page_ids(
            free_index,
            page_size=self.page_size,
        )
        owner_page_start, owner_page_end = self._get_free_page_owner_bounds()
        self._debug_validate_free_page_blocks(
            free_index,
            free_page_indices,
            page_size=self.page_size,
            owner_page_start=owner_page_start,
            owner_page_end=owner_page_end,
        )
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def _get_free_page_owner_bounds(self) -> tuple[int, int]:
        return 1, self.num_pages + 1

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )
