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


from typing import Optional

import torch

from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    get_local_dp_buffer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_parallel


def maybe_prefetch_next_full_attention_kv(
    forward_batch: ForwardBatch,
    next_full_attention_layer_id: Optional[int],
) -> None:
    """Prefetch (owner-broadcast) the next layer's DSA KV under layer split.

    No-op unless the current batch runs DSA prefill-CP and the active KV pool is
    a layer-sharded pool exposing ``prefetch_kv_buffer`` (i.e.
    ``LayerSplitDSATokenToKVPool``). Kicking the broadcast off one layer ahead
    overlaps it with the current layer's attention compute.
    """
    if next_full_attention_layer_id is None or not dsa_use_prefill_cp(forward_batch):
        return

    prefetch_kv_buffer = getattr(get_token_to_kv_pool(), "prefetch_kv_buffer", None)
    if prefetch_kv_buffer is not None:
        prefetch_kv_buffer(next_full_attention_layer_id)


def dsa_cp_gather_hidden_states(hidden_states: torch.Tensor):
    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_dp_size == 1 and attn_tp_size == 1
    hidden_states, local_hidden_states = (
        get_local_dp_buffer(get_parallel().attn_cp_group),
        hidden_states,
    )
    attn_cp_all_gather_into_tensor(hidden_states, local_hidden_states)
    return hidden_states


def dsa_cp_reduce_scatter_hidden_states(hidden_states: torch.Tensor):
    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_dp_size == 1 and attn_tp_size == 1
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    input_hidden_states = hidden_states
    hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]
    attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
    return hidden_states
