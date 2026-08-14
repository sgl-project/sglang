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

from sglang.kernels.ops.npu import get_npu_vector_core_count


@triton.jit(do_not_specialize=["batch_size", "topk"])
def _build_full_tree_kernel(
    parent_list_ptr,
    selected_index_ptr,
    verified_seq_len_ptr,
    seq_len_prefix_sum_ptr,
    tree_mask_ptr,
    positions_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    batch_size,
    topk,
    PARENT_STRIDE: tl.constexpr,
    SELECTED_STRIDE: tl.constexpr,
    DRAFT_TOKEN_NUM: tl.constexpr,
    BLOCK_DRAFT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_DRAFT)
    offsets_i64 = offsets.to(tl.int64)
    row_mask = offsets < DRAFT_TOKEN_NUM

    for batch_idx in tl.range(pid, batch_size, num_programs):
        batch_i64 = batch_idx.to(tl.int64)
        token_base = batch_i64 * DRAFT_TOKEN_NUM
        parent_base = batch_i64 * PARENT_STRIDE
        selected_base = batch_i64 * SELECTED_STRIDE
        seq_len = tl.load(verified_seq_len_ptr + batch_i64)
        prefix_sum = tl.load(seq_len_prefix_sum_ptr + batch_i64)
        tree_base = token_base * DRAFT_TOKEN_NUM + prefix_sum * DRAFT_TOKEN_NUM

        tl.store(
            retrieve_index_ptr + token_base + offsets_i64,
            token_base + offsets_i64,
            mask=row_mask,
        )
        tl.store(
            retrieve_next_token_ptr + token_base + offsets_i64,
            -1,
            mask=row_mask,
        )
        tl.store(
            retrieve_next_sibling_ptr + token_base + offsets_i64,
            -1,
            mask=row_mask,
        )
        tl.store(positions_ptr + token_base, seq_len)

        for token_idx in range(DRAFT_TOKEN_NUM - 1, 0, -1):
            parent_tb_idx = (
                tl.load(selected_index_ptr + selected_base + token_idx - 1) // topk
            )
            parent_position = 0
            if parent_tb_idx > 0:
                parent_token_idx = tl.load(
                    parent_list_ptr + parent_base + parent_tb_idx
                )
                parent_position = DRAFT_TOKEN_NUM
                for candidate_pos in range(DRAFT_TOKEN_NUM - 1):
                    selected = tl.load(
                        selected_index_ptr + selected_base + candidate_pos
                    )
                    if (
                        parent_position == DRAFT_TOKEN_NUM
                        and selected == parent_token_idx
                    ):
                        parent_position = candidate_pos + 1

            if parent_position != DRAFT_TOKEN_NUM:
                next_ptr = retrieve_next_token_ptr + token_base + parent_position
                previous_child = tl.load(next_ptr)
                tl.store(next_ptr, token_idx)
                if previous_child != -1:
                    tl.store(
                        retrieve_next_sibling_ptr + token_base + token_idx,
                        previous_child,
                    )

        for token_idx in range(DRAFT_TOKEN_NUM):
            token_tree_base = tree_base + (seq_len + DRAFT_TOKEN_NUM) * token_idx
            tl.store(
                tree_mask_ptr + token_tree_base + offsets_i64,
                offsets == 0,
                mask=row_mask,
            )
            if token_idx > 0:
                position = 0
                current = token_idx - 1
                active = 1
                for _ in range(DRAFT_TOKEN_NUM):
                    if active == 1:
                        position += 1
                        tl.store(tree_mask_ptr + token_tree_base + current + 1, True)
                        parent_tb_idx = (
                            tl.load(selected_index_ptr + selected_base + current) // topk
                        )
                        if parent_tb_idx == 0:
                            active = 0
                        else:
                            parent_token_idx = tl.load(
                                parent_list_ptr + parent_base + parent_tb_idx
                            )
                            next_position = DRAFT_TOKEN_NUM - 1
                            for candidate_pos in range(DRAFT_TOKEN_NUM - 1):
                                selected = tl.load(
                                    selected_index_ptr + selected_base + candidate_pos
                                )
                                if (
                                    next_position == DRAFT_TOKEN_NUM - 1
                                    and selected == parent_token_idx
                                ):
                                    next_position = candidate_pos
                            current = next_position
                tl.store(positions_ptr + token_base + token_idx, seq_len + position)


def build_full_tree_npu(
    *,
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    topk: int,
    draft_token_num: int,
) -> None:
    """Build the FULL_MASK EAGLE tree with the A5 vector-core kernel."""
    if parent_list.dim() != 2 or selected_index.dim() != 2:
        raise ValueError("A5 tree inputs must be rank-2 tensors")
    batch_size = int(verified_seq_len.numel())
    if batch_size == 0:
        return
    seq_len_prefix_sum = torch.cumsum(verified_seq_len, dim=0) - verified_seq_len
    num_cores = min(get_npu_vector_core_count(), batch_size)
    _build_full_tree_kernel[(num_cores,)](
        parent_list,
        selected_index,
        verified_seq_len,
        seq_len_prefix_sum,
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        batch_size,
        topk,
        PARENT_STRIDE=parent_list.stride(0),
        SELECTED_STRIDE=selected_index.stride(0),
        DRAFT_TOKEN_NUM=draft_token_num,
        BLOCK_DRAFT=triton.next_power_of_2(draft_token_num),
    )
