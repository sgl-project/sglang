# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def alloc_extend_npu_kernel(
    prefix_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    BS_UPPER: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    max_num_extend_tokens,
    BLOCK_SIZE: tl.constexpr = 2048,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BS_UPPER)
    seq_lens = tl.load(seq_lens_ptr + offsets, mask=offsets <= pid, other=0)
    prefix_lens = tl.load(prefix_lens_ptr + offsets, mask=offsets <= pid, other=0)
    extend_lens = seq_lens - prefix_lens
    seq_len = tl.load(seq_lens_ptr + pid)
    prefix_len = tl.load(prefix_lens_ptr + pid)
    extend_len = seq_len - prefix_len
    output_start = tl.sum(extend_lens) - extend_len

    pages_after = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
    pages_before = (prefix_lens + PAGE_SIZE - 1) // PAGE_SIZE
    num_new_pages = pages_after - pages_before
    num_pages_for_row = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE - (
        prefix_len + PAGE_SIZE - 1
    ) // PAGE_SIZE
    new_page_start = tl.sum(num_new_pages) - num_pages_for_row

    last_loc = tl.load(last_loc_ptr + pid).to(tl.int64)
    num_part1 = (
        min(seq_len, (prefix_len + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE) - prefix_len
    )
    page_offsets = tl.arange(0, PAGE_SIZE)
    tl.store(
        out_indices + output_start + page_offsets,
        last_loc + 1 + page_offsets,
        mask=page_offsets < num_part1,
    )
    if prefix_len + num_part1 == seq_len:
        return

    num_part2 = (
        seq_len // PAGE_SIZE * PAGE_SIZE
        - (prefix_len + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
    )
    block_offsets = tl.arange(0, BLOCK_SIZE)
    for block_idx in range(tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)):
        current = block_offsets + block_idx * BLOCK_SIZE
        page_start = tl.load(
            free_page_ptr + new_page_start + current // PAGE_SIZE,
            mask=current < num_part2,
        )
        tl.store(
            out_indices + output_start + num_part1 + current,
            page_start * PAGE_SIZE + current % PAGE_SIZE,
            mask=current < num_part2,
        )
    if prefix_len + num_part1 + num_part2 == seq_len:
        return

    num_part3 = seq_len - seq_len // PAGE_SIZE * PAGE_SIZE
    start_loc = tl.load(free_page_ptr + new_page_start + num_pages_for_row - 1)
    tl.store(
        out_indices + output_start + num_part1 + num_part2 + page_offsets,
        start_loc * PAGE_SIZE + page_offsets,
        mask=page_offsets < num_part3,
    )


def alloc_extend_npu(
    *,
    prefix_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    last_loc: torch.Tensor,
    free_pages: torch.Tensor,
    out_indices: torch.Tensor,
    page_size: int,
    max_num_extend_tokens: int,
) -> None:
    batch_size = int(prefix_lens.shape[0])
    if batch_size == 0:
        return
    alloc_extend_npu_kernel[(batch_size,)](
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        BS_UPPER=triton.next_power_of_2(batch_size),
        PAGE_SIZE=page_size,
        max_num_extend_tokens=max_num_extend_tokens,
    )
