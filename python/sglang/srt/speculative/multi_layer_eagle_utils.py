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

from sglang.kernels.ops.speculative.multi_layer_eagle import (
    compute_widened_draft_extend_locs_positions_triton,
    fill_draft_extend_prepare_buffers_triton,
    fill_widened_draft_extend_inputs_triton,
    rotate_input_ids,
    rotate_input_ids_kernel,
    stash_append_boundary_state_triton,
    wide_row_softmax_triton,
)
from sglang.srt.environ import envs


def boundary_kv_fix_enabled() -> bool:
    return envs.SGLANG_ENABLE_MTP_BOUNDARY_KV_FIX.get()


def compute_widened_draft_extend_locs_positions(
    seq_lens,
    req_pool_indices,
    req_to_token,
    stash_valid_lens,
    draft_token_num: int,
    num_front_tokens: int,
    num_warmup_tokens: int,
):
    """Batched out_cache_loc + positions for the widened draft-extend window,
    from pre-verify state. Invalid/warm-up front rows write to sacrificial loc 0."""
    return compute_widened_draft_extend_locs_positions_triton(
        seq_lens,
        req_pool_indices,
        req_to_token,
        stash_valid_lens,
        draft_token_num,
        num_front_tokens,
        num_warmup_tokens,
    )


__all__ = [
    "boundary_kv_fix_enabled",
    "compute_widened_draft_extend_locs_positions",
    "fill_draft_extend_prepare_buffers_triton",
    "fill_widened_draft_extend_inputs_triton",
    "rotate_input_ids",
    "rotate_input_ids_kernel",
    "stash_append_boundary_state_triton",
    "wide_row_softmax_triton",
]
