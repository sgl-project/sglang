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

import torch
import triton
import triton.language as tl
from typing import List, Optional, Tuple
import heapq

from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.utils import get_bool_env_var, next_power_of_2

def alloc_extend_kernel_native(
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    last_locs: torch.Tensor,
    free_page: torch.Tensor,
    out_indices: torch.Tensor,
    ret_values: torch.Tensor,
    page_size: int,
):
    sum_num_new_pages = 0
    sum_extend_lens = 0
    for i in range(len(prefix_lens)):
        pre_len = prefix_lens[i]
        seq_len = seq_lens[i]
        last_loc = last_locs[i]

        extend_len = seq_len - pre_len
        sum_extend_lens += seq_len - pre_len
        output_start_loc = sum_extend_lens - extend_len

        num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
            pre_len + page_size - 1
        ) // page_size
        sum_num_new_pages += num_page_start_loc_self
        new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

        num_part1 = min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
        assert num_part1 >= 0 and num_part1 <= page_size
        offset_part1 = torch.arange(0, num_part1)
        out_indices[output_start_loc + offset_part1] = last_loc + 1 + offset_part1

        if pre_len + num_part1 == seq_len:
            continue

        num_part2 = seq_len // page_size * page_size - (pre_len + page_size - 1) // page_size * page_size
        offset_part2 = torch.arange(0, num_part2)
        page_start = free_page[new_page_start_loc + offset_part2 // page_size]
        out_indices[output_start_loc + num_part1 + offset_part2] = page_start * page_size + offset_part2 % page_size

        if pre_len + num_part1 + num_part2 == seq_len:
            continue

        num_part3 = seq_len - seq_len // page_size * page_size
        start_loc = free_page[new_page_start_loc + num_page_start_loc_self - 1]
        offset_part3 = torch.arange(0, num_part3)
        out_indices[output_start_loc + num_part1 + num_part2 + offset_part3] = start_loc * page_size + offset_part3
    ret_values.fill_((sum_num_new_pages << 32) | sum_extend_lens)


def alloc_decode_kernel_native(
    seq_lens: torch.Tensor,
    last_locs: torch.Tensor,
    free_page: torch.Tensor,
    out_indices: torch.Tensor,
    ret_values: torch.Tensor,
    page_size: int,
):
    sum_num_new_pages = 0
    for i in range(len(seq_lens)):
        seq_len = seq_lens[i]
        pre_len = seq_len - 1
        last_loc = last_locs[i]

        num_pages_after = (seq_len + page_size - 1) // page_size
        num_pages_before = (pre_len + page_size - 1) // page_size
        num_new_pages = num_pages_after - num_pages_before

        num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
            pre_len + page_size - 1
        ) // page_size
        sum_num_new_pages += num_new_pages
        new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

        if num_page_start_loc_self == 0:
            out_indices[i] = last_loc + 1
        else:
            page = free_page[new_page_start_loc]
            out_indices[i] = page * page_size
    ret_values.fill_(sum_num_new_pages)

    
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
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

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
        merged_value = (sum_num_new_pages.to(tl.int64)) << 32 | sum_extend_lens.to(
            tl.int64
        )
        tl.store(ret_values, merged_value)

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )

    offset_many_page = tl.arange(0, max_num_extend_tokens)
    page_start = tl.load(
        free_page_ptr + new_page_start_loc + offset_many_page // page_size,
        mask=offset_many_page < num_part2,
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + offset_many_page,
        page_start * page_size + offset_many_page % page_size,
        mask=offset_many_page < num_part2,
    )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
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


class PagedTokenToKVPoolAllocator:
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
        self.size = size
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.num_pages = size // page_size

        self.free_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")

        self._kvcache = kvcache
        self.ret_values = torch.empty((), dtype=torch.int64, device=self.device)

    def available_size(self):
        return len(self.free_pages) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
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

        bs = len(prefix_lens)
        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )
        if self.device != "cpu":
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
        else:
            alloc_extend_kernel_native(
                prefix_lens,
                seq_lens,
                last_loc,
                self.free_pages,
                out_indices,
                self.ret_values,
                self.page_size,
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

        bs = len(seq_lens)
        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        if self.device != "cpu":
            alloc_decode_kernel[(bs,)](
                seq_lens,
                last_loc,
                self.free_pages,
                out_indices,
                self.ret_values,
                next_power_of_2(bs),
                self.page_size,
                )
        else:
            alloc_decode_kernel_native(
                seq_lens,
                last_loc,
                self.free_pages,
                out_indices,
                self.ret_values,
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
            self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def backup_state(self):
        return self.free_pages

    def restore_state(self, free_pages):
        self.free_pages = free_pages

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []


class BlockManager():
    def __init__(self, block_size, num_blocks):
        self.block_size = block_size
        self.num_blocks = num_blocks
        # Use a heap for free blocks to always get the smallest block ID
        self.free_blocks = list(range(1, num_blocks))
        heapq.heapify(self.free_blocks)
        self.req_to_block_ids = {}
        # Track sequence info: (block_ids, used_slots_in_last_block, slot_ids)
        self.seq_info = {}

    def clear(self):
        self.free_blocks = list(range(1, self.num_blocks))
        heapq.heapify(self.free_blocks)
        self.req_to_block_ids = {}
        self.seq_info = {}

    def available_size(self):
        return len(self.free_blocks) * self.block_size

    def allocate(self, req_id, need_size: int) -> Optional[List[int]]:
        """Allocate slots for a sequence.

        Args:
            req_id: Request ID for the sequence
            need_size: Number of tokens needed for the sequence

        Returns:
            List of slot IDs, or None if allocation failed
        """
        remaining_tokens = need_size
        allocated_slots = []
        new_blocks = []

        # If sequence exists, try to fill remaining space in last block
        if req_id in self.seq_info:
            block_ids, used_slots, slot_ids = self.seq_info[req_id]
            last_block = block_ids[-1]
            available_in_last = self.block_size - used_slots

            tokens_to_add = min(remaining_tokens, available_in_last)
            if tokens_to_add > 0:
                start_slot = last_block * self.block_size + used_slots
                new_slots = list(range(start_slot, start_slot + tokens_to_add))
                allocated_slots.extend(new_slots)
                remaining_tokens -= tokens_to_add
                used_slots += tokens_to_add
                self.seq_info[req_id] = (block_ids, used_slots, slot_ids + new_slots)

        # Need new blocks
        if remaining_tokens > 0:
            blocks_needed = (remaining_tokens + self.block_size - 1) // self.block_size

            if blocks_needed > len(self.free_blocks):
                return None

            # Get the smallest blocks using heap
            new_blocks = [heapq.heappop(self.free_blocks) for _ in range(blocks_needed)]

            # Calculate slots in new blocks
            for i, block_id in enumerate(new_blocks):
                tokens_in_block = min(self.block_size, remaining_tokens)
                start_slot = block_id * self.block_size
                new_slots = list(range(start_slot, start_slot + tokens_in_block))
                allocated_slots.extend(new_slots)
                remaining_tokens -= tokens_in_block

                # Update sequence info
                if req_id in self.seq_info:
                    existing_blocks, _, existing_slot_ids = self.seq_info[req_id]
                    all_blocks = existing_blocks + [block_id]
                    all_slot_ids = existing_slot_ids + new_slots
                else:
                    all_blocks = [block_id]
                    all_slot_ids = new_slots
                self.seq_info[req_id] = (all_blocks, tokens_in_block, all_slot_ids)

            # Update block mapping
            if req_id in self.req_to_block_ids:
                self.req_to_block_ids[req_id].extend(new_blocks)
            else:
                self.req_to_block_ids[req_id] = new_blocks

        return allocated_slots, new_blocks

    def free(self, req_id):
        """Free all blocks allocated to a request.

        Args:
            req_id: Request ID whose blocks should be freed
        """
        if req_id in self.req_to_block_ids:
            freed_blocks = self.req_to_block_ids[req_id]
            # Push freed blocks back into the heap
            for block in freed_blocks:
                heapq.heappush(self.free_blocks, block)
            del self.req_to_block_ids[req_id]
            del self.seq_info[req_id]

class HPUPagedTokenToKVPoolAllocator:

    def __init__(self, size, page_size, dtype, device, kvcache):
        self.block_manager = BlockManager(page_size, size // page_size)
        self.kvcache = kvcache
        self.size = size
        self.dtype = dtype
        self.device = device
        self.page_size = page_size

    def available_size(self):
        return self.block_manager.available_size()

    def alloc_extend(self, id_len_pairs: List[Tuple[int, int]]):
        slot_ids = []
        for seq_id, need_size in id_len_pairs:
            slots, new_blocks = self.block_manager.allocate(seq_id, need_size)
            # print(f"seq_id: {seq_id}, need_size: {need_size}, slots: {slots}")
            # print(f"block_manager.seq_info: {self.block_manager.seq_info}")
            slot_ids.extend(slots)
        return slot_ids

    def alloc_decode(self, seq_ids: List[int]):
        slot_ids = []
        for seq_id in seq_ids:
            slots, new_blocks = self.block_manager.allocate(seq_id, 1)
            # print(f"seq_id: {seq_id}, slots: {slots}")
            # print(f"block_manager.seq_info: {self.block_manager.seq_info}")
            slot_ids.extend(slots)
        return slot_ids

    def free(self, req_id: int):
        self.block_manager.free(req_id)

    def free_group_begin(self):
        pass

    def free_group_end(self):
        pass

    def clear(self):
        self.block_manager.clear()

