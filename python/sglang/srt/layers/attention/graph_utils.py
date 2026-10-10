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

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.model_executor.forward_context import get_attn_backend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


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


def _new_attention_output(
    layer: RadixAttention, q: torch.Tensor, v: Optional[torch.Tensor]
) -> torch.Tensor:
    """An uninitialized buffer for the layer's attention output over q's rows.

    Its dtype follows v (the model dtype) when available: qk-norm may emit q in
    a different dtype without changing the dtype the backend writes. FP8 q/v
    (e.g. mxfp8 KV-cache attention) still produce a bf16 attention output;
    sizing the buffer off an fp8 dtype would silently cast-copy the result to
    fp8.
    """
    out_dtype = v.dtype if v is not None else q.dtype
    if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        out_dtype = torch.bfloat16
    if layer.qk_head_dim != layer.v_head_dim:
        return q.new_empty(
            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim), dtype=out_dtype
        )
    return torch.empty_like(q, dtype=out_dtype)


# Per-row attention inputs beside Q, K and V that the tc-piecewise op schema
# cannot carry. The padded paths narrow them with the rows they belong to.
_PER_ROW_EXTRA_KWARGS = (
    "rel_bias",
    "q_descale",
    "k_descale",
    "v_descale",
    "mxfp8_norm_rope_positions",
    "mxfp8_norm_rope_temp_scale",
)
# The per-row inputs that follow K and V rather than the queries.
_KEY_ROW_KWARGS = frozenset({"k_rope", "k_descale", "v_descale", "idx_k", "idx_v"})


def padded_extend_real_tokens(
    q: torch.Tensor, forward_batch: ForwardBatch
) -> Optional[int]:
    """The number of real rows of an extend batch that MLP sync padded to a multiple of
    attention TP, whose attention metadata covers only those rows; None for a
    batch without such padding. Target verify plans its padded rows itself."""
    mode = forward_batch.forward_mode
    if not mode.is_extend() or mode.is_target_verify():
        return None
    real_num_tokens = forward_batch.global_num_token_non_padded_cpu
    if real_num_tokens is None or not 0 < real_num_tokens < q.shape[0]:
        return None
    return real_num_tokens


def _attention_on_real_rows(
    layer: RadixAttention,
    real_num_tokens: int,
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    save_kv_cache: bool,
    *,
    key_value_num_tokens: Optional[int] = None,
    **kwargs,
):
    """Run the backend on a padded batch's real rows, with the cache locations
    and positions narrowed for the call, and return its output padded back
    with zero rows, as the prefill graph path does. K and V, and the per-row
    inputs that follow them, are narrowed with the queries unless the caller
    gives their own extent (K and V that hold a cached prefix), the layer is a
    cross-attention (K and V are the encoder's), or they have a different row
    count. A sparse indexer's q, k and v go with the rows they
    index. A backend that writes into forward_batch._attn_output fills the
    padded output's real rows in place."""
    num_tokens = q.shape[0]

    def queries(t):
        return t[:real_num_tokens] if t is not None else None

    keys_own_rows = key_value_num_tokens is not None or layer.is_cross_attention

    def keys(t):
        if t is None or keys_own_rows or t.shape[0] != num_tokens:
            return t
        return t[:real_num_tokens]

    for name in (
        "q_rope",
        "k_rope",
        "topk_indices",
        "idx_q",
        "idx_k",
        "idx_v",
        *_PER_ROW_EXTRA_KWARGS,
    ):
        if kwargs.get(name) is not None:
            narrow = keys if name in _KEY_ROW_KWARGS else queries
            kwargs[name] = narrow(kwargs[name])
    if kwargs.get("aux_tensors") is not None:
        kwargs["aux_tensors"] = [queries(t) for t in kwargs["aux_tensors"]]
    # A backend that returns more than the output (its LSE, or a sparse
    # indexer's output beside it) returns a tuple, padded as it comes.
    output = (
        None
        if kwargs.get("return_lse")
        or forward_batch.mha_return_lse
        or kwargs.get("idx_q") is not None
        else _new_attention_output(layer, q, v)
    )
    out_cache_loc = forward_batch.out_cache_loc
    positions = forward_batch.positions
    attn_output = forward_batch._attn_output
    forward_batch.out_cache_loc = out_cache_loc[:real_num_tokens]
    if positions is not None:
        forward_batch.positions = positions[:real_num_tokens]
    if output is not None:
        forward_batch._attn_output = output[:real_num_tokens]
    try:
        ret = get_attn_backend().forward(
            queries(q),
            keys(k),
            keys(v),
            layer,
            forward_batch,
            save_kv_cache,
            **kwargs,
        )
    finally:
        forward_batch.out_cache_loc = out_cache_loc
        forward_batch.positions = positions
        forward_batch._attn_output = attn_output

    def padded(t):
        if not isinstance(t, torch.Tensor) or t.shape[0] != real_num_tokens:
            return t
        full = t.new_empty((num_tokens, *t.shape[1:]))
        full[:real_num_tokens].copy_(t)
        full[real_num_tokens:].zero_()
        return full

    if isinstance(ret, tuple):
        return tuple(padded(t) for t in ret)
    if (
        output is None
        or not isinstance(ret, torch.Tensor)
        or ret.shape[0] != real_num_tokens
        or ret.numel() != output[:real_num_tokens].numel()
    ):
        return padded(ret)
    if ret.data_ptr() != output.data_ptr():
        output[:real_num_tokens].view(ret.shape).copy_(ret)
    output[real_num_tokens:].zero_()
    return output.view(num_tokens, *ret.shape[1:])
