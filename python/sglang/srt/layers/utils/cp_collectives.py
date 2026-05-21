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

"""Low-level CP collective primitives shared by both strategies.

These wrap the ``attn_cp`` NCCL group with the same padding/unpadding contract
that ``cp_utils.py`` used to provide. Strategy implementations in
``cp_strategy.py`` call into these; backends should not import them directly."""

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


def cp_all_gather_reorganized_into_tensor(
    input_tensor, total_len, cp_size, forward_batch, stream
):
    """Allgather + remove-padding for ``hidden_states``-shaped tensors
    (2-D ``[seq, hidden]``)."""
    max_len = (total_len + cp_size - 1) // cp_size
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        input_tensor = F.pad(
            input_tensor, (0, 0, 0, pad_size), mode="constant", value=0
        )
    with use_symmetric_memory(
        get_attention_cp_group(), disabled=not is_allocation_symmetric()
    ):
        full = torch.empty(
            max_len * cp_size,
            input_tensor.shape[1],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    get_attention_cp_group().cp_all_gather_into_tensor_async(full, input_tensor, stream)

    outputs_list_max = list(
        torch.split(full, forward_batch.attn_cp_metadata.max_rank_len, dim=0)
    )
    return torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )


def cp_all_gather_reorganized_into_tensor_kv_cache(
    input_tensor, total_len, cp_size, forward_batch, stream
):
    """Allgather + remove-padding for KV-cache-shaped tensors
    (``[seq, num_heads, head_dim]`` or other multi-dim)."""
    max_len = (total_len + cp_size - 1) // cp_size
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        padding = [0, 0] * (input_tensor.ndim - 1) + [0, pad_size]
        input_tensor = F.pad(input_tensor, padding, mode="constant", value=0)

    with use_symmetric_memory(
        get_attention_cp_group(), disabled=not is_allocation_symmetric()
    ):
        full = torch.empty(
            max_len * cp_size,
            *input_tensor.shape[1:],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

    get_attention_cp_group().cp_all_gather_into_tensor_async(full, input_tensor, stream)

    outputs_list_max = list(
        torch.split(full, forward_batch.attn_cp_metadata.max_rank_len, dim=0)
    )
    return torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )
