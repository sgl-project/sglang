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

"""Low-level CP allgather primitives.

These are the padding-aware allgather + remove-padding routines used by the
zigzag (in-seq-split) CP path. They are kept here as free functions (ported
verbatim from the pre-refactor ``layers/utils/cp_utils.py``) so the strategy
classes and the per-layer communicator can share one implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    get_attention_cp_group,
    is_allocation_symmetric,
)


def cp_all_gather_reorganized_into_tensor(input_tensor, cp_size, forward_batch, stream):
    """
    Allgather communication for context_parallel(kv_cache, index_k, hidden_states).
    This implementation mainly consists of three parts:
    Step 1, padding the input shape to unify the shape for allgather communication (the shape must be the same).
    Step 2, allgather communication(async).
    Step 3, removing the padding and reassembling the data according to the actual tokens.
    """
    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        input_tensor = F.pad(
            input_tensor, (0, 0, 0, pad_size), mode="constant", value=0
        )
    with use_symmetric_memory(
        get_attention_cp_group(), disabled=not is_allocation_symmetric()
    ):
        input_tensor_full = torch.empty(
            max_len * cp_size,
            input_tensor.shape[1],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    get_attention_cp_group().cp_all_gather_into_tensor_async(
        input_tensor_full, input_tensor, stream
    )

    outputs_list_max = list(
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs


def cp_all_gather_reorganized_into_tensor_kv_cache(
    input_tensor, cp_size, forward_batch, stream
):
    """
    Allgather communication for context_parallel KV cache.
    Handles multi-dimensional tensors (e.g., [seq_len, num_heads, head_dim]).
    """
    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        # Pad the first dimension (seq_len). F.pad expects padding in reverse dimension order.
        # For n dimensional tensor, we need 2*n values: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
        # To pad only the first dimension: [0, 0] * (ndim - 1) + [0, pad_size]
        padding = [0, 0] * (input_tensor.ndim - 1) + [0, pad_size]
        input_tensor = F.pad(input_tensor, padding, mode="constant", value=0)

    # Create output tensor with proper shape for all dimensions
    with use_symmetric_memory(
        get_attention_cp_group(), disabled=not is_allocation_symmetric()
    ):
        input_tensor_full = torch.empty(
            max_len * cp_size,
            *input_tensor.shape[1:],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    get_attention_cp_group().cp_all_gather_into_tensor_async(
        input_tensor_full, input_tensor, stream
    )

    outputs_list_max = list(
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs
