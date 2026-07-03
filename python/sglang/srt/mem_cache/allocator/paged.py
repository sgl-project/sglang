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
import triton

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.triton_ops.allocator import (
    alloc_decode_kernel,
    alloc_extend_kernel,
    free_dual_pool_kernel,
)
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
    num_new_pages = (seq_lens + page_size - 1) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    num_full_new_pages = (seq_lens) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    need_page = num_new_pages - num_full_new_pages
    end_new_pages = torch.cumsum(num_new_pages, 0)
    start_new_pages = end_new_pages - num_new_pages
    pos_in_page = torch.arange(page_size, device=device, dtype=torch.int32)
    for i in range(len(prefix_lens)):
        num1 = (
            min(
                seq_lens[i],
                (prefix_lens[i] + page_size - 1) // page_size * page_size,
            )
            - prefix_lens[i]
        )
        if num1:
            out_indices[start_pos[i] : start_pos[i] + num1] = (
                last_loc[i] + 1 + pos_in_page[:num1].view(-1)
            )

        if prefix_lens[i] + num1 == seq_lens[i]:
            continue

        num2 = (
            seq_lens[i] // page_size - (prefix_lens[i] + page_size - 1) // page_size
        ) * page_size
        if num2:
            pages = (
                free_pages[start_new_pages[i] : end_new_pages[i] - need_page[i]]
                * page_size
            )
            out_indices[start_pos[i] + num1 : start_pos[i] + num1 + num2] = (
                pages.view(-1, 1) + pos_in_page.view(1, -1)
            ).view(-1)

        if prefix_lens[i] + num1 + num2 == seq_lens[i]:
            continue

        num3 = seq_lens[i] - seq_lens[i] // page_size * page_size
        if num3:
            out_indices[end_pos[i] - num3 : end_pos[i]] = (
                free_pages[end_new_pages[i] - 1] * page_size + pos_in_page[:num3]
            ).view(-1)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.

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

        # Pre-warm the torch.unique HIP kernel used in free(). When a request
        # finishes with a prompt that already exists in the radix tree (e.g.
        # bench_serving sending the same warmup+measured prompt), the radix
        # cache's _insert_helper frees the duplicate KV indices via
        # token_to_kv_pool_allocator.free(value[start:prefix_len]). That call
        # path runs `torch.unique(free_index // self.page_size)` on a
        # ~prompt_len-sized int64 tensor. The first such call on AMD ROCm
        # JIT-compiles rocPRIM sort/unique kernels and costs ~200ms, which
        # shows up as a mysterious "second-request slow" (Run 1) for
        # repeated-prompt benchmarks. Running it once at init time moves
        # that JIT cost to startup. This is a ROCm-only JIT cost, so the
        # warm-up is gated on _is_hip and skipped on other platforms.
        if _is_hip and torch.cuda.is_available():
            try:
                _warmup = torch.arange(1024, dtype=torch.int64, device=device)
                _ = torch.unique(_warmup // page_size)
                torch.cuda.synchronize()
            except Exception:
                pass
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
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

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


class DeviceFreeListPagedAllocator(PagedTokenToKVPoolAllocator):
    """Paged allocator whose free list lives on device (ring buffer plus
    device head/tail counters).

    Frees append to the ring inside free_dual_pool_kernel (single launch,
    page-granular dedup via epoch election), so the free path never forces a
    CPU-GPU sync even when the live-page count is data-dependent (the SWA
    mapping side). The host tracks occupancy as (tail snapshot - allocs):
    allocs are host-driven and exact, the tail snapshot arrives through an
    async pinned copy and is re-adopted with one event wait only when queried
    after frees.
    """

    def __init__(self, size, page_size, dtype, device, kvcache, need_sort):
        num_pages = size // page_size
        # Ring capacity num_pages + 1 so legitimate occupancy never wraps.
        self._cap = num_pages + 1
        self._buf = torch.empty(self._cap, dtype=torch.int64, device=device)
        # Absolute (never wrapping) counters; physical index = abs % _cap.
        self._head = torch.zeros((), dtype=torch.int64, device=device)
        self._tail = torch.zeros((), dtype=torch.int64, device=device)
        self._tail_pinned = torch.zeros((), dtype=torch.int64, pin_memory=True)
        self._free_event = torch.cuda.Event()
        # Winner-election epochs (page-granular free dedup); monotone, one
        # counter per array, never cleared.
        self._page_epoch = torch.zeros(num_pages + 2, dtype=torch.int32, device=device)
        self._cur_epoch = 0
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)

    def clear(self):
        n = self.num_pages = self.size // self.page_size
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self._buf[:n] = torch.arange(1, n + 1, dtype=torch.int64, device=self.device)
        self._head.zero_()
        self._tail.fill_(n)
        self._head_host = 0
        self._tail_snapshot = n
        self._dirty = False
        self.is_not_in_free_group = True
        self.free_group = []

    def _refresh(self):
        # One event wait covers all frees enqueued so far; no-op when clean.
        if self._dirty:
            self._free_event.synchronize()
            self._tail_snapshot = int(self._tail_pinned)
            self._dirty = False
            occupancy = self._tail_snapshot - self._head_host
            assert occupancy <= self.num_pages, (
                f"free list overflow (double free): occupancy {occupancy} > "
                f"{self.num_pages} pages"
            )

    def available_size(self):
        self._refresh()
        return (self._tail_snapshot - self._head_host) * self.page_size

    def _pages_available(self, num_pages: int) -> bool:
        if self._tail_snapshot - self._head_host >= num_pages:
            return True
        self._refresh()
        return self._tail_snapshot - self._head_host >= num_pages

    def _consume(self, num_pages: int):
        idx = (self._head + torch.arange(num_pages, device=self.device)) % self._cap
        pages = self._buf[idx]
        self._head += num_pages
        self._head_host += num_pages
        return pages

    def _mark_freed(self):
        self._tail_pinned.copy_(self._tail, non_blocking=True)
        self._free_event.record()
        self._dirty = True

    def free(self, free_index: torch.Tensor):
        # Page-granular dedup free of arbitrary indices, same semantics as the
        # legacy torch.unique path but one kernel launch and sync-free.
        if free_index.numel() == 0:
            return
        if not self.is_not_in_free_group:
            self.free_group.append(free_index)
            return
        if free_index.dtype != torch.int64:
            free_index = free_index.to(torch.int64)
        self._cur_epoch += 1
        n = free_index.numel()
        BLOCK = 256
        free_dual_pool_kernel[(triton.cdiv(n, BLOCK),)](
            free_index,
            n,
            self._page_epoch,
            self._cur_epoch,
            self._buf,
            self._cap,
            self._tail,
            # SCAN_SWA off: the swa-side args are unused (dead code)
            self._buf,
            self._page_epoch,
            0,
            self._buf,
            1,
            self._tail,
            page_size=self.page_size,
            BLOCK=BLOCK,
            MARK_SELF=True,
            SCAN_SWA=False,
        )
        self._mark_freed()

    def alloc(self, need_size: int):
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"
        num_pages = need_size // self.page_size
        if not self._pages_available(num_pages):
            return None
        pages = self._consume(num_pages)
        return (
            pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

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
        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        if not self._pages_available(num_new_pages):
            return None

        bs = len(prefix_lens)
        consumed = self._consume(num_new_pages)
        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )
        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            consumed,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )
        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)
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
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if not self._pages_available(num_new_pages):
            return None

        bs = len(seq_lens)
        consumed = self._consume(num_new_pages)
        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            consumed,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )
        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)
        return out_indices

    def merge_and_sort_free(self):
        # The ring mixes sorted and freshly freed pages; sort valid entries
        # in place (invalid ones to the end via a max sentinel) and rebase.
        occ = self._tail - self._head
        j = torch.arange(self._cap, device=self.device)
        valid = (j - self._head) % self._cap < occ
        sentinel = torch.iinfo(torch.int64).max
        vals = torch.where(valid, self._buf[: self._cap], torch.full_like(j, sentinel))
        vals, _ = torch.sort(vals)
        self._buf[: self._cap] = vals
        self._tail.copy_(occ)
        self._head.zero_()
        self._head_host = 0
        self._tail_pinned.copy_(self._tail, non_blocking=True)
        self._free_event.record()
        self._dirty = True

    def backup_state(self):
        return (
            self._buf.clone(),
            self._head.clone(),
            self._tail.clone(),
            self._head_host,
        )

    def restore_state(self, state):
        buf, head, tail, head_host = state
        self._buf.copy_(buf)
        self._head.copy_(head)
        self._tail.copy_(tail)
        self._head_host = head_host
        # A pinned copy from an undone free may still land; re-snapshot the
        # restored tail behind it so the next refresh adopts the right value.
        self._tail_pinned.copy_(self._tail, non_blocking=True)
        self._free_event.record()
        self._dirty = True
