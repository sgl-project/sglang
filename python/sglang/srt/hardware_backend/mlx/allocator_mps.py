"""Paged KV cache allocator for Apple Metal (MPS).

``PagedTokenToKVPoolAllocator`` allocates page-aligned KV slots with two Triton
kernels (``alloc_extend_kernel`` / ``alloc_decode_kernel``). Triton has no Metal
backend, so on MPS those launches raise ``TypeError: 'function' object is not
subscriptable`` and any ``--page-size > 1`` run dies during prefill.

This subclass replaces both with pure-torch equivalents, following the same
pattern ``NPUPagedTokenToKVPoolAllocator`` uses for Ascend. Only the index math
is overridden -- the free-page bookkeeping, ``free()``, and the KV cache itself
are inherited unchanged, so slots stay page-aligned and contiguous exactly as
the block paged attention kernel requires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.utils import get_num_new_pages

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class MpsPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    """Triton-free paged allocator used on MPS when ``page_size > 1``."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: "KVCache",
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)

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

        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        # Checked before writing any indices: alloc_extend_naive indexes
        # free_pages directly and would read past the end otherwise.
        if num_new_pages > len(self.free_pages):
            return None

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )
        alloc_extend_naive(
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            self.page_size,
            self.device,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

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

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        # A request opens a new page exactly when its new token is the first
        # slot of one, i.e. seq_len % page_size == 1. Everyone else continues
        # into last_loc + 1.
        need_new_pages = (seq_lens % self.page_size == 1).int()
        end_new_pages = torch.cumsum(need_new_pages, 0)
        start_new_pages = end_new_pages - need_new_pages
        if num_new_pages == 0:
            out_indices = last_loc + 1
        else:
            out_indices = (last_loc + 1) * (1 - need_new_pages) + self.free_pages[
                start_new_pages
            ] * self.page_size * need_new_pages

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices.to(torch.int64)
