"""Torch-MPS allocation of contiguous physical KV pages."""

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.utils import get_num_new_pages


class MPSPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
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
        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        if num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None
        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )
        alloc_extend_naive(
            prefix_lens=prefix_lens,
            seq_lens=seq_lens,
            last_loc=last_loc,
            free_pages=self.free_pages,
            out_indices=out_indices,
            page_size=self.page_size,
            device=self.device,
        )
        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
        )
        if num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None
        if num_new_pages == 0:
            return last_loc.to(torch.int64) + 1
        needs_page = seq_lens % self.page_size == 1
        page_indices = (torch.cumsum(needs_page.long(), dim=0) - 1).clamp(min=0)
        out_indices = torch.where(
            needs_page, self.free_pages[page_indices] * self.page_size, last_loc + 1
        )
        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices
