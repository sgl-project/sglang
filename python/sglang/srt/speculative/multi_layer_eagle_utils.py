# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

import torch

from sglang.srt.utils import is_cpu

_is_cpu = is_cpu()

if _is_cpu:
    from sgl_kernel import rotate_input_ids_cpu

if not _is_cpu:
    from sglang.srt.speculative.triton_ops.multi_layer_eagle import (
        assign_hidden_states_pool_triton,
        rotate_input_ids_kernel,
        rotate_input_ids_triton,
    )
else:
    rotate_input_ids_kernel = None
    rotate_input_ids_triton = None


def rotate_input_ids(
    input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index=None
):
    if _is_cpu:
        # rotate_input_ids_cpu mutates input_ids in place (callers rely on
        # this) and requires int64 tensors; extend_* may arrive int32.
        rotate_input_ids_cpu(
            input_ids,
            extend_start_loc.to(torch.int64),
            extend_seq_lens.to(torch.int64),
            topk_index.to(torch.int64),
            select_index.to(torch.int64) if select_index is not None else None,
        )
        return input_ids
    return rotate_input_ids_triton(
        input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index
    )


def assign_hidden_states_pool(
    hidden_states: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
    pool_size: int,
    num_seqs: int,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
):
    if _is_cpu:
        for req in range(num_seqs):
            pool_idx = req_pool_indices[req]
            extend_len = extend_seq_lens[req]
            start_loc = extend_start_loc[req]
            end_loc = start_loc + extend_len
            req_to_hidden_states_pool[pool_idx, :pool_size, :].copy_(
                hidden_states[end_loc - pool_size : end_loc, :]
            )
    else:
        assign_hidden_states_pool_triton(
            hidden_states,
            req_pool_indices,
            req_to_hidden_states_pool,
            pool_size,
            num_seqs,
            extend_seq_lens,
            extend_start_loc,
        )


__all__ = [
    "rotate_input_ids_kernel",
    "rotate_input_ids_triton",
    "assign_hidden_states_pool",
    "rotate_input_ids",
]
