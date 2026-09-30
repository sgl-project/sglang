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
"""Token slicing and temporary batch state for graph attention calls."""

from contextlib import contextmanager

import torch


def _zero_padded_tokens(output, actual_tokens):
    if actual_tokens is not None:
        output[actual_tokens:].zero_()


@contextmanager
def attention_input_scope(forward_batch, output, query_tokens, kv_tokens, kwargs):
    """Narrow per-token operands and restore the caller's batch even on failure."""
    kwargs = dict(kwargs)
    for name in (
        "q_rope",
        "topk_indices",
        "rel_bias",
        "q_descale",
        "idx_q",
        "mxfp8_norm_rope_positions",
        "mxfp8_norm_rope_temp_scale",
    ):
        if kwargs.get(name) is not None:
            kwargs[name] = kwargs[name][:query_tokens]
    for name in ("k_rope", "k_descale", "v_descale", "idx_k", "idx_v"):
        if kwargs.get(name) is not None:
            kwargs[name] = kwargs[name][:kv_tokens]
    if kwargs.get("aux_tensors") is not None:
        kwargs["aux_tensors"] = [
            tensor[:query_tokens] for tensor in kwargs["aux_tensors"]
        ]
    cache_loc, positions = forward_batch.out_cache_loc, forward_batch.positions
    previous_output = forward_batch._attn_output
    forward_batch.out_cache_loc = cache_loc[:query_tokens]
    if positions is not None:
        forward_batch.positions = positions[:query_tokens]
    forward_batch._attn_output = output[:query_tokens]
    try:
        yield kwargs
    finally:
        forward_batch.out_cache_loc, forward_batch.positions = cache_loc, positions
        forward_batch._attn_output = previous_output


def allocate_attention_outputs(layer, query, value, index_query):
    """Allocate graph-owned buffers before entering the eager attention region."""
    rows = query.shape[0]
    sparse = index_query is not None
    dtype = query.dtype if sparse or value is None else value.dtype
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        dtype = torch.bfloat16
    shape = (
        (rows, layer.tp_q_head_num * layer.v_head_dim)
        if sparse or layer.qk_head_dim != layer.v_head_dim
        else query.shape
    )
    output = torch.empty(shape, dtype=dtype, device=query.device)
    index_output = (
        query.new_empty((rows, index_query.shape[1] * index_query.shape[2]))
        if sparse
        else None
    )
    return output, index_output
