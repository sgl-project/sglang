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
    from sgl_kernel import rotate_input_ids_cpu as _rotate_input_ids_cpp
else:
    from sglang.srt.speculative.triton_ops.multi_layer_eagle import (
        assign_hidden_states_pool_triton as _assign_hidden_states_pool_triton_gpu,
        rotate_input_ids_triton as _rotate_input_ids_triton_gpu,
    )

from sglang.srt.speculative.triton_ops.multi_layer_eagle import (
    assign_hidden_states_pool_torch,
    assign_new_state_triton,
)


def rotate_input_ids(
    input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index=None
):
    if _is_cpu:
        # _rotate_input_ids_cpp requires int64 tensors; input_ids is mutated in
        # place and may be int32, so rotate an int64 copy and write it back.
        if extend_start_loc.dtype != torch.int64:
            extend_start_loc = extend_start_loc.to(torch.int64)
        if extend_seq_lens.dtype != torch.int64:
            extend_seq_lens = extend_seq_lens.to(torch.int64)
        if topk_index.dtype != torch.int64:
            topk_index = topk_index.to(torch.int64)
        if select_index is not None and select_index.dtype != torch.int64:
            select_index = select_index.to(torch.int64)
        input_ids64 = (
            input_ids if input_ids.dtype == torch.int64 else input_ids.to(torch.int64)
        )
        _rotate_input_ids_cpp(
            input_ids64, extend_start_loc, extend_seq_lens, topk_index, select_index
        )
        if input_ids64 is not input_ids:
            input_ids.copy_(input_ids64)
    else:
        return _rotate_input_ids_triton_gpu(
            input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index
        )
    return input_ids


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
        assign_hidden_states_pool_torch(
            hidden_states,
            req_pool_indices,
            req_to_hidden_states_pool,
            pool_size,
            num_seqs,
            extend_seq_lens,
            extend_start_loc,
        )
    else:
        _assign_hidden_states_pool_triton_gpu(
            hidden_states,
            req_pool_indices,
            req_to_hidden_states_pool,
            pool_size,
            num_seqs,
            extend_seq_lens,
            extend_start_loc,
        )


__all__ = [
    "assign_hidden_states_pool_torch",
    "assign_hidden_states_pool",
    "assign_new_state_triton",
    "rotate_input_ids",
]
