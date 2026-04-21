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
# ==============================================================================
"""Attention entry point for the breakable CUDA graph (BCG) runner.

:func:`bcg_unified_attention_with_output` mirrors
:func:`sglang.srt.layers.radix_attention.unified_attention_with_output` exactly
at the call site; internally it installs a graph break so the attention call
runs eagerly at every replay, and delegates the common path back to
``unified_attention_with_output``. BCG only adds an MLA-model MHA-chunk shape
fixup on top.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.layers.radix_attention import unified_attention_with_output

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


__all__ = ["bcg_unified_attention_with_output"]


def bcg_unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> None:
    """BCG counterpart of :func:`unified_attention_with_output`."""
    global _breakable_attention_fn
    if _breakable_attention_fn is None:
        from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import eager_on_graph

        _breakable_attention_fn = eager_on_graph(True)(_bcg_attention_body)
    _breakable_attention_fn(
        query, key, value, output, save_kv_cache, layer_id,
        q_rope=q_rope, k_rope=k_rope, sinks=sinks,
    )


# Lazy-init: wrapped with eager_on_graph(True) on first use to avoid
# import-time dependency on breakable_cuda_graph (which requires cuda.bindings).
_breakable_attention_fn = None


def _bcg_attention_body(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    *,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> None:
    """Body invoked at every replay of the BCG attention break point.

    Non-MLA-chunk path delegates to PCG's ``unified_attention_with_output`` so
    the backend-call / output-copy logic is shared.
    """
    context = get_forward_context()
    forward_batch = context.forward_batch

    n = _mla_dispatch_to_mha(forward_batch, save_kv_cache, k_rope, query)
    if n is None:
        unified_attention_with_output(
            query, key, value, output, save_kv_cache, layer_id,
            q_rope=q_rope, k_rope=k_rope, sinks=sinks,
        )
        return

    # MLA-model MHA chunk path: needs real-size slicing and chunked-prefix
    # merge. ``attention_layers[layer_id]`` stores attn_mqa (the MLA absorbed
    # layer); attn_mha is reachable via the sibling pointer stashed by the
    # MLA model's forward method.
    attention_layer = context.attention_layers[layer_id]._bcg_mha_layer
    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope
    if k_rope is not None:
        kwargs["k_rope"] = k_rope
    if sinks is not None:
        kwargs["sinks"] = sinks
    ret = _bcg_mha_attention(
        query, key, value, attention_layer, forward_batch, save_kv_cache, n, kwargs
    )
    output[:n].view(ret.shape).copy_(ret)


def _mla_dispatch_to_mha(
    forward_batch: "ForwardBatch",
    save_kv_cache: bool,
    k_rope: Optional[torch.Tensor],
    query: torch.Tensor,
) -> Optional[int]:
    """Return extend_num_tokens if this is an MLA-model MHA layer that needs
    capture-size → real-size slicing at replay, else None.

    MHA layers (save_kv_cache=False, no k_rope) receive static capture-size
    tensors but the MHA backend expects real extend-size tensors.
    """
    n = getattr(forward_batch, "_bcg_mha_slice_n", None)
    if (
        n is not None
        and not save_kv_cache
        and k_rope is None
        and (query.shape[0] > n or forward_batch._bcg_mha_has_prefix)
    ):
        return n
    return None


def _bcg_mha_attention(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    attention_layer: "RadixAttention",
    forward_batch: "ForwardBatch",
    save_kv_cache: bool,
    n: int,
    kwargs: dict,
) -> torch.Tensor:
    """MHA attention for BCG replay in MLA models.

    Slices q/k/v to extend_num_tokens, runs extend-only attention, then runs
    the chunked prefix attention loop (which doesn't replay in CUDA graphs
    since it's Python code between break points).
    """
    query = query[:n]
    if key is not None:
        key = key[:n]
    if value is not None:
        value = value[:n]

    has_prefix = forward_batch._bcg_mha_has_prefix

    # Set flags for extend-only attention, request LSE if prefix exists.
    # mha_one_shot must be False: with True, FA3 uses cu_seqlens_k that
    # includes prefix length, but k only has extend tokens.
    forward_batch.mha_return_lse = has_prefix
    forward_batch.mha_one_shot = False
    forward_batch.set_attn_attend_prefix_cache(False)

    ret = forward_batch.attn_backend.forward(
        query, key, value, attention_layer, forward_batch, save_kv_cache, **kwargs
    )

    if has_prefix and hasattr(attention_layer, "_bcg_chunked_prefix_fn"):
        attn_output, lse = ret
        forward_batch.set_attn_attend_prefix_cache(True)
        ret = attention_layer._bcg_chunked_prefix_fn(
            q=query.view(
                -1, attention_layer.tp_q_head_num, attention_layer.qk_head_dim
            ),
            accum_output=attn_output,
            accum_lse=lse,
            forward_batch=forward_batch,
        )

    forward_batch.set_attn_attend_prefix_cache(False)
    forward_batch.mha_return_lse = False
    forward_batch.mha_one_shot = True

    return ret
