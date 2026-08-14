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


def build_full_tree_mask_reference(
    *,
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    topk: int,
    draft_token_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for the flattened FULL_MASK layout and positions."""
    batch_size = int(verified_seq_len.numel())
    total_mask_size = sum(
        (int(seq_len) + draft_token_num) * draft_token_num
        for seq_len in verified_seq_len
    )
    tree_mask = torch.ones(
        total_mask_size,
        dtype=torch.bool,
        device=verified_seq_len.device,
    )
    positions = torch.empty(
        batch_size * draft_token_num,
        dtype=torch.long,
        device=verified_seq_len.device,
    )

    tree_base = 0
    for batch_idx in range(batch_size):
        seq_len = int(verified_seq_len[batch_idx])
        selected = selected_index[batch_idx, : draft_token_num - 1]
        for token_idx in range(draft_token_num):
            row_base = tree_base + token_idx * (seq_len + draft_token_num)
            draft_base = row_base + seq_len
            tree_mask[draft_base : draft_base + draft_token_num] = False
            tree_mask[draft_base] = True

            position = 0
            current = token_idx - 1
            if token_idx > 0:
                for _ in range(draft_token_num):
                    position += 1
                    tree_mask[draft_base + current + 1] = True
                    parent_tb_idx = int(selected[current]) // topk
                    if parent_tb_idx == 0:
                        break
                    parent_token_idx = parent_list[batch_idx, parent_tb_idx]
                    matches = torch.nonzero(selected == parent_token_idx).flatten()
                    if matches.numel() == 0:
                        break
                    current = int(matches[0])

            positions[batch_idx * draft_token_num + token_idx] = seq_len + position
        tree_base += (seq_len + draft_token_num) * draft_token_num

    return tree_mask, positions


def read_cache_locations_reference(
    *,
    req_pool_indices: torch.Tensor,
    token_pool: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
) -> torch.Tensor:
    """Reference for the ragged cache-location gather used by the A5 kernel."""
    chunks = []
    for row_idx in range(req_pool_indices.numel()):
        req_idx = int(req_pool_indices[row_idx])
        start = int(start_offset[row_idx])
        end = int(end_offset[row_idx])
        chunks.append(token_pool[req_idx, start:end])
    if not chunks:
        return token_pool.new_empty((0,))
    return torch.cat(chunks)


def build_retrieval_links_reference(
    *,
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    topk: int,
    draft_token_num: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for EAGLE retrieve, child, and sibling link construction."""
    batch_size = parent_list.shape[0]
    retrieve_index = torch.arange(
        batch_size * draft_token_num,
        dtype=torch.long,
        device=parent_list.device,
    ).view(batch_size, draft_token_num)
    next_token = torch.full_like(retrieve_index, -1)
    next_sibling = torch.full_like(retrieve_index, -1)

    for batch_idx in range(batch_size):
        selected = selected_index[batch_idx, : draft_token_num - 1]
        for token_idx in range(draft_token_num - 1, 0, -1):
            parent_tb_idx = int(selected[token_idx - 1]) // topk
            parent_position = 0
            if parent_tb_idx > 0:
                parent_token_idx = parent_list[batch_idx, parent_tb_idx]
                matches = torch.nonzero(selected == parent_token_idx).flatten()
                if matches.numel() == 0:
                    continue
                parent_position = int(matches[0]) + 1

            previous_child = int(next_token[batch_idx, parent_position])
            next_token[batch_idx, parent_position] = token_idx
            if previous_child != -1:
                next_sibling[batch_idx, token_idx] = previous_child

    return retrieve_index, next_token, next_sibling
