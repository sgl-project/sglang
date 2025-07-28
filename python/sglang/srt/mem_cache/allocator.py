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

"""
Page-aligned memory pool.
"""

import abc
import weakref
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.memory_pool import SWAKVPool
from sglang.srt.utils import get_bool_env_var, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

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

    def get_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(self, size: int, dtype: torch.dtype, device: str, kvcache: KVCache):
        super().__init__(size, 1, dtype, device, kvcache)
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int):
        if need_size > len(self.free_pages):
            self.merge_and_sort_free()
        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            self.release_pages = torch.cat((self.release_pages, free_index))
        else:
            self.free_group.append(free_index)

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        dtype: torch.dtype,
        device: str,
        kvcache: SWAKVPool,
    ):
        super().__init__(size, 1, dtype, device, kvcache)
        assert isinstance(kvcache, SWAKVPool)
        self._size_full = size
        self._size_swa = size_swa
        self.full_attn_allocator = TokenToKVPoolAllocator(
            size,
            dtype,
            device,
            kvcache.full_kv_pool,
        )
        self.swa_attn_allocator = TokenToKVPoolAllocator(
            size_swa,
            dtype,
            device,
            kvcache.swa_kv_pool,
        )
        self.full_to_swa_index_mapping = torch.empty(
            size + size_swa + 1,
            dtype=torch.int64,
            device=device,
        )
        self.clear()

        self._kvcache.full_to_swa_index_mapping = self.full_to_swa_index_mapping

    def available_size(self):
        raise NotImplementedError()

    def full_available_size(self):
        return self.full_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    @property
    def size_full(self):
        return self._size_full

    @property
    def size_swa(self):
        return self._size_swa

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def alloc(self, need_size: int):
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index)
            self.free_swa(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def free_swa(self, free_index: torch.Tensor):
        swa_indices = self.full_to_swa_index_mapping[free_index]
        swa_indices = swa_indices[swa_indices > 0]
        self.swa_attn_allocator.free(swa_indices)
        self.full_to_swa_index_mapping[free_index] = 0

    def backup_state(self):
        raise NotImplementedError

    def restore_state(self, state):
        raise NotImplementedError

    def clear(self):
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        self.full_to_swa_index_mapping.fill_(0)
        self.is_in_free_group = False
        self.free_group = []


@triton.jit
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    ret_values,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
    max_num_extend_tokens: tl.constexpr,
):
    """
    Triton kernel for efficient page-aligned token location allocation during sequence extension.
    
    This kernel handles the complex task of allocating token locations for extending sequences
    while maintaining page alignment. It implements a three-part allocation strategy to handle
    different scenarios efficiently:
    
    1. **Part 1 - Partial Page Continuation**: Reuses existing page space for tokens that
       fit within the current page boundary
    2. **Part 2 - Full Page Allocation**: Allocates complete new pages for tokens that
       span multiple pages
    3. **Part 3 - Final Partial Page**: Allocates one additional page for remaining tokens
       that don't fill a complete page
    
    The kernel processes each request in parallel and ensures that:
    - Token locations are page-aligned for efficient memory access
    - Memory fragmentation is minimized by reusing existing page space
    - Allocation is atomic and thread-safe across multiple requests
    
    Args:
        pre_lens_ptr: Pointer to array containing prefix lengths for each request.
                     Shape: [batch_size]
                     pre_lens[i] = number of existing tokens for request i
        seq_lens_ptr: Pointer to array containing new sequence lengths for each request.
                     Shape: [batch_size]
                     seq_lens[i] = total number of tokens for request i after extension
        last_loc_ptr: Pointer to array containing the last allocated token location for each request.
                     Shape: [batch_size]
                     last_loc[i] = last token location allocated for request i
        free_page_ptr: Pointer to array of available page indices in the free page pool.
                      Shape: [num_free_pages]
                      Contains page indices that can be allocated
        out_indices: Pointer to output array where allocated token locations will be stored.
                    Shape: [total_extend_tokens]
                    Output: flat list of allocated token locations
        ret_values: Pointer to return value array for communication with host.
                   Shape: [1]
                   Stores: (num_new_pages << 32) | total_extend_tokens
        bs_upper: Upper bound for batch size (next power of 2 for efficient processing)
        page_size: Number of tokens per page (e.g., 64 for FlashMLA, 128 for CutlassMLA)
        max_num_extend_tokens: Maximum number of tokens being extended (next power of 2)
    
    Example:
        If we have:
        - pre_lens = [5, 3] (request 0 has 5 existing tokens, request 1 has 3)
        - seq_lens = [8, 6] (request 0 needs 8 total tokens, request 1 needs 6)
        - page_size = 4
        - last_loc = [19, 11] (last allocated locations)
        
        The kernel will:
        1. For request 0: reuse locations [20, 21] from existing page, allocate new page for [24, 25, 26, 27]
        2. For request 1: reuse location [12] from existing page, allocate new page for [16, 17, 18]
        
        Output: out_indices = [20, 21, 24, 25, 26, 27, 12, 16, 17, 18]
    """
    # Get the program ID - each program processes one request in the batch
    pid = tl.program_id(0)

    # Step 1: Load sequence information for all requests up to current one
    # This is needed to calculate cumulative offsets for output positioning
    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens  # Calculate how many new tokens each request needs

    # Step 2: Load current request's specific information
    seq_len = tl.load(seq_lens_ptr + pid)  # Total tokens needed for this request
    pre_len = tl.load(pre_lens_ptr + pid)  # Existing tokens for this request
    extend_len = seq_len - pre_len         # New tokens needed for this request

    # Step 3: Calculate output positioning for this request
    # We need to know where in the output array to write this request's token locations
    sum_extend_lens = tl.sum(extend_lens)  # Total tokens being extended across all requests
    output_start_loc = sum_extend_lens - extend_len  # Start position for this request's output

    # Step 4: Calculate page requirements for all requests
    # This determines how many new pages need to be allocated
    num_pages_after = (seq_lens + page_size - 1) // page_size   # Pages needed after extension
    num_pages_before = (pre_lens + page_size - 1) // page_size  # Pages needed before extension
    num_new_pages = num_pages_after - num_pages_before          # New pages needed

    # Step 5: Calculate page allocation positioning for this specific request
    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size  # New pages needed for this request
    sum_num_new_pages = tl.sum(num_new_pages)  # Total new pages needed across all requests
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self  # Start position in free page pool

    # Step 6: Return aggregated information to host
    # Only the last program writes the return value to avoid race conditions
    if pid == tl.num_programs(0) - 1:
        # Pack two values into one: (num_new_pages << 32) | total_extend_tokens
        merged_value = (sum_num_new_pages.to(tl.int64)) << 32 | sum_extend_lens.to(
            tl.int64
        )
        tl.store(ret_values, merged_value)

    # ============================================================================
    # PART 1: Fill the old partial page (reuse existing page space)
    # ============================================================================
    # This handles tokens that can fit within the existing page boundary
    # We reuse the page that was already allocated for the previous tokens
    
    last_loc = tl.load(last_loc_ptr + pid)  # Get the last allocated location for this request
    
    # Calculate how many tokens can fit in the existing page
    # This is the minimum of:
    # 1. Total tokens needed (seq_len)
    # 2. End of current page boundary ((pre_len + page_size - 1) // page_size * page_size)
    # Minus the existing tokens (pre_len)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    
    # Create offsets within the page [0, 1, 2, ..., page_size-1]
    offset_one_page = tl.arange(0, page_size)
    
    # Store token locations by continuing from the last allocated location
    # Each token gets location: last_loc + 1 + offset
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,  # Only store valid tokens
    )
    
    # Early exit if all tokens fit in the existing page
    if pre_len + num_part1 == seq_len:
        return

    # ============================================================================
    # PART 2: Fill the new full pages (allocate complete new pages)
    # ============================================================================
    # This handles tokens that span multiple complete pages
    # We allocate new pages from the free page pool
    
    # Calculate how many tokens need complete new pages
    # This is the number of tokens from the end of the first page to the start of the last page
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )

    # Create offsets for all potential tokens [0, 1, 2, ..., max_num_extend_tokens-1]
    offset_many_page = tl.arange(0, max_num_extend_tokens)
    
    # Load page indices from the free page pool
    # Each token needs a page index: new_page_start_loc + (token_offset // page_size)
    page_start = tl.load(
        free_page_ptr + new_page_start_loc + offset_many_page // page_size,
        mask=offset_many_page < num_part2,  # Only load for valid tokens
    )
    
    # Store token locations for complete pages
    # Each token gets location: page_index * page_size + offset_within_page
    tl.store(
        out_indices + output_start_loc + num_part1 + offset_many_page,
        page_start * page_size + offset_many_page % page_size,
        mask=offset_many_page < num_part2,  # Only store valid tokens
    )
    
    # Early exit if all remaining tokens fit in complete pages
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # ============================================================================
    # PART 3: Fill the new partial page (allocate one more page for remaining tokens)
    # ============================================================================
    # This handles the final tokens that don't fill a complete page
    # We allocate one additional page for these remaining tokens
    
    # Calculate how many tokens remain (those that don't fill complete pages)
    num_part3 = seq_len - seq_len // page_size * page_size
    
    # Load the page index for the final partial page
    # This is the last page we need to allocate for this request
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    
    # Store token locations for the final partial page
    # Each token gets location: page_index * page_size + offset_within_page
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,  # Only store valid tokens
    )


@triton.jit
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    ret_values,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Return value
    if pid == tl.num_programs(0) - 1:
        tl.store(ret_values, sum_num_new_pages)

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)


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
    ):
        super().__init__(size, page_size, dtype, device, kvcache)
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")
        self.ret_values = torch.empty((), dtype=torch.int64, device=self.device)
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

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
        seq_lens: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        estimated_num_new_pages = (
            (
                (seq_lens + self.page_size - 1) // self.page_size
                - (prefix_lens + self.page_size - 1) // self.page_size
            )
            .sum()
            .item()
        )
        if estimated_num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()

        bs = len(prefix_lens)
        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )
        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            self.ret_values,
            next_power_of_2(bs),
            self.page_size,
            next_power_of_2(extend_num_tokens),
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        merged_value = self.ret_values.item()
        num_new_pages = merged_value >> 32
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        estimated_num_new_pages = (
            (
                (seq_lens + self.page_size - 1) // self.page_size
                - (seq_lens - 1 + self.page_size - 1) // self.page_size
            )
            .sum()
            .item()
        )
        if estimated_num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()

        bs = len(seq_lens)
        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            self.ret_values,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = self.ret_values.item()
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            self.release_pages = torch.cat((free_page_indices, self.release_pages))
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

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


def alloc_extend_kernel_ascend(
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

        num3 = seq_lens[i] - seq_lens[i] // page_size * page_size
        if num3:
            out_indices[end_pos[i] - num3 : end_pos[i]] = (
                free_pages[end_new_pages[i] - 1] * page_size + pos_in_page[:num3]
            ).view(-1)
    return num_new_pages


def alloc_decode_kernel_ascend(
    seq_lens,
    last_loc,
    free_pages,
    out_indices,
    page_size,
):
    num_new_pages = (seq_lens + page_size - 1) // page_size - (
        seq_lens - 1 + page_size - 1
    ) // page_size
    end_new_pages = torch.cumsum(num_new_pages, 0)
    start_new_pages = end_new_pages - num_new_pages
    for i in range(len(seq_lens)):
        if num_new_pages[i]:
            out_indices[i] = free_pages[start_new_pages[i]] * page_size
        else:
            out_indices[i] = last_loc[i] + 1
    return num_new_pages


class AscendPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
    ):
        super().__init__(size, page_size, dtype, device, kvcache)
        self.ret_values = torch.empty((), dtype=torch.int32, device=self.device)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        estimated_num_new_pages = (
            (
                (seq_lens + self.page_size - 1) // self.page_size
                - (prefix_lens + self.page_size - 1) // self.page_size
            )
            .sum()
            .item()
        )
        if estimated_num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()

        bs = len(prefix_lens)
        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int32, device=self.device
        )

        self.ret_values = alloc_extend_kernel_ascend(
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

        num_new_pages = self.ret_values.sum()
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        estimated_num_new_pages = (
            (
                (seq_lens + self.page_size - 1) // self.page_size
                - (seq_lens - 1 + self.page_size - 1) // self.page_size
            )
            .sum()
            .item()
        )
        if estimated_num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()

        bs = len(seq_lens)
        out_indices = torch.empty((bs,), dtype=torch.int32, device=self.device)

        self.ret_values = alloc_decode_kernel_ascend(
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = self.ret_values.sum()
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def clear(self):
        super().clear()
        self.free_pages = self.free_pages.to(torch.int32)
        self.release_pages = self.release_pages.to(torch.int32)
