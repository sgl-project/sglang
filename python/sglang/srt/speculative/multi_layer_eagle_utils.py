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
    from sgl_kernel import assign_new_state_cpu as _assign_new_state_cpp
    from sgl_kernel import rotate_input_ids_cpu as _rotate_input_ids_cpp

if not _is_cpu:
    from sglang.srt.speculative.triton_ops.multi_layer_eagle import (
        assign_hidden_states_pool_triton as _assign_hidden_states_pool_triton_gpu,
        assign_new_state_triton as _assign_new_state_triton_gpu,
        rotate_input_ids_triton as _rotate_input_ids_triton_gpu,
    )

from sglang.srt.speculative.triton_ops.multi_layer_eagle import (
    assign_hidden_states_pool_torch,
)


def rotate_input_ids(
    input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index=None
):
    if _is_cpu:
        _rotate_input_ids_cpp(
            input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index
        )
    else:
        return _rotate_input_ids_triton_gpu(
            input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index
        )
    return input_ids


def assign_new_state(
    next_token_ids: torch.Tensor,
    old_input_ids: torch.Tensor,
    old_positions: torch.Tensor,
    old_hidden_states: torch.Tensor,
    old_out_cache_loc: torch.Tensor,
    old_extend_seq_lens: torch.Tensor,
    old_extend_start_loc: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    padding_lens: torch.Tensor,
    num_seqs: int,
    step: int,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
):
    """
    Wrapper function to calculate offsets and launch the Triton or CPU kernel.
    """
    if _is_cpu:
        _assign_new_state_cpp(
            next_token_ids,
            old_input_ids,
            old_positions,
            old_out_cache_loc,
            old_extend_seq_lens,
            old_extend_start_loc,
            input_ids,
            positions,
            out_cache_loc,
            extend_seq_lens,
            extend_start_loc,
            seq_lens,
            padding_lens,
            req_pool_indices,
            req_to_token,
            num_seqs,
            step,
            hidden_states,
            old_hidden_states,
            req_to_hidden_states_pool,
        )
    else:
        _assign_new_state_triton_gpu(
            next_token_ids,
            old_input_ids,
            old_positions,
            old_hidden_states,
            old_out_cache_loc,
            old_extend_seq_lens,
            old_extend_start_loc,
            input_ids,
            positions,
            hidden_states,
            out_cache_loc,
            extend_seq_lens,
            extend_start_loc,
            seq_lens,
            padding_lens,
            num_seqs,
            step,
            req_pool_indices,
            req_to_token,
            req_to_hidden_states_pool,
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
    "assign_new_state",
    "rotate_input_ids",
]
