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

from sglang.kernels.ops.memory.allocator import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.utils import (
    get_bool_env_var,
    get_num_new_pages,
    is_hip,
    next_power_of_2,
)

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


def alloc_extend_naive(
    prefix_lens,
    seq_lens,
    last_loc,
    free_pages,
    out_indices,
    page_size,
    device,
):
    extend_lens = seq_lens - prefix_lens
    end_pos = torch.cumsum(extend_lens, 0)
    start_pos = end_pos - extend_lens

    extend_num_tokens = out_indices.shape[0]
    if extend_num_tokens == 0:
        return

    j = torch.arange(extend_num_tokens, device=device, dtype=torch.int64)
    owner = torch.searchsorted(end_pos, j, right=True)
    local = j - start_pos[owner]
    last_loc_g = last_loc[owner]

    ceil_prefix = (prefix_lens + page_size - 1) // page_size * page_size
    floor_seq = seq_lens // page_size * page_size

    if free_pages.numel() == 0:
        # Only valid when no request needs a new page; nothing below indexes
        # the empty pool, so a short pool would silently return garbage.
        ceil_seq = (seq_lens + page_size - 1) // page_size * page_size
        assert torch.all(ceil_seq == ceil_prefix), (
            "alloc_extend_naive: free_pages is empty but the batch requires "
            "new pages; caller must ensure pool >= demand"
        )
        out_indices.copy_(last_loc_g + 1 + local)
        return

    num1 = torch.clamp(seq_lens, max=ceil_prefix) - prefix_lens
    done_after_1 = (prefix_lens + num1) == seq_lens
    num2 = torch.where(done_after_1, torch.zeros_like(num1), floor_seq - ceil_prefix)
    num3 = torch.where(done_after_1, torch.zeros_like(num1), seq_lens - floor_seq)

    full_pages = num2 // page_size
    need_extra_page = (num3 > 0).to(torch.int64)
    pages_per_req = full_pages + need_extra_page
    end_new_pages = torch.cumsum(pages_per_req, 0)
    start_new_pages = end_new_pages - pages_per_req

    num1_g = num1[owner]
    num2_g = num2[owner]
    start_new_pages_g = start_new_pages[owner]
    end_new_pages_g = end_new_pages[owner]

    is_phase1 = local < num1_g
    is_phase2 = (~is_phase1) & (local < num1_g + num2_g)

    val_phase1 = last_loc_g + 1 + local

    rel2 = torch.clamp(local - num1_g, min=0)
    # torch.where below evaluates both branches per slot, so dead-lane page
    # indices must stay in range even where their phase is never selected.
    page_idx2 = torch.clamp(
        start_new_pages_g + rel2 // page_size, min=0, max=free_pages.numel() - 1
    )
    pos_in_page2 = rel2 % page_size
    val_phase2 = free_pages[page_idx2] * page_size + pos_in_page2

    rel3 = torch.clamp(local - num1_g - num2_g, min=0)
    page_idx3 = torch.clamp(end_new_pages_g - 1, min=0, max=free_pages.numel() - 1)
    val_phase3 = free_pages[page_idx3] * page_size + rel3

    out = torch.where(
        is_phase1, val_phase1, torch.where(is_phase2, val_phase2, val_phase3)
    )
    out_indices.copy_(out)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Same interface as `TokenToKVPoolAllocator`, but the indices handed to one
    request are always page-aligned.

    TODO: fuse last_loc into the kernel.
    """

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

        # Pre-warm the torch.unique used by free(): on ROCm the first call
        # JIT-compiles rocPRIM sort/unique kernels and costs ~200ms.
        if _is_hip and torch.cuda.is_available():
            try:
                _warmup = torch.arange(1024, dtype=torch.int64, device=device)
                _ = torch.unique(_warmup // page_size)
                torch.cuda.synchronize()
            except Exception:
                pass
        self.clear()

    def available_size(self):
        return (len(self.free_pages) + self.num_staged_pages) * self.page_size

    def get_all_free_pages(self):
        return torch.cat((self.free_pages, *self.staged_pages))

    def merge_and_sort_free(self):
        if not self.staged_pages:
            return
        self.free_pages, _ = torch.sort(self.get_all_free_pages())
        self.staged_pages = []
        self.num_staged_pages = 0

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert need_size % self.page_size == 0, (
                "The allocation size should be page-aligned"
            )

        num_pages = need_size // self.page_size
        if num_pages > len(self.free_pages):
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
        if extend_num_tokens // self.page_size + bs + 1 > len(self.free_pages):
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
        if bs > len(self.free_pages):
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
        if free_index.numel() == 0:
            return

        if self.free_group is None:
            self._release_page_ids(torch.unique(free_index // self.page_size))
        else:
            self.free_group.append(self._copy_for_free_group(free_index))

        if self.debug_mode:
            self._debug_check_no_duplicate_pages()

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        """Fixed-shape free(): page-aligned start plus contiguous per-page tokens
        make ``free_index[::page_size]`` hit each page once; no torch.unique sync."""
        if free_index.numel() == 0:
            return

        ps = self.page_size
        assert start_pos % ps == 0, f"segment start {start_pos} is not page-aligned"
        reps = free_index[::ps]

        if self.debug_mode:
            # reference unique on CPU: the NPU subclass deliberately avoids device unique
            assert torch.equal(
                torch.sort(reps.cpu() // ps)[0],
                torch.unique(free_index.cpu() // ps),
            )

        if self.free_group is None:
            self._release_page_ids(reps // ps)
            if self.debug_mode:
                self._debug_check_no_duplicate_pages()
        else:
            self.free_page_reps_group.append(self._copy_for_free_group(reps))

    def _debug_check_no_duplicate_pages(self):
        pages = self.get_all_free_pages()
        assert len(torch.unique(pages)) == len(pages)

    def _release_page_ids(self, *page_ids: torch.Tensor):
        if self.need_sort:
            self.staged_pages.extend(page_ids)
            self.num_staged_pages += sum(ids.numel() for ids in page_ids)
        else:
            self.free_pages = torch.cat((*page_ids, self.free_pages))

    def free_group_begin(self):
        super().free_group_begin()
        self.free_page_reps_group = []

    def free_group_end(self):
        super().free_group_end()
        if self.free_page_reps_group:
            self._release_page_ids(
                torch.cat(self.free_page_reps_group) // self.page_size
            )
            self.free_page_reps_group = []
        if self.debug_mode:
            # the no-double-free contract can only break across a group's calls
            self._debug_check_no_duplicate_pages()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.free_group = None
        self.free_page_reps_group = []
        # need_sort only: freed pages wait here, unsorted, until an alloc runs short.
        self.staged_pages: list[torch.Tensor] = []
        self.num_staged_pages = 0

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
