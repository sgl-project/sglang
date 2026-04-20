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
"""Radix attention."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class AttentionType(Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Decoder bidirectional attention between image tokens
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"


class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        pos_encoding_mode: str = "NONE",
        logit_capping_method: str = "tanh",
        quant_config: Optional[QuantizationConfig] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.sliding_window_size = sliding_window_size or -1
        self.is_cross_attention = is_cross_attention
        self.use_irope = use_irope
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = None
        self.v_scale_float = None
        self.quant_method = None

        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is not None:
            self.quant_method.create_weights(self)
        self.attn_type = attn_type

        self.pos_encoding_mode = pos_encoding_mode
        self.logit_capping_method = logit_capping_method
        self.xai_temperature_len = -1

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
            if get_global_server_args().enable_breakable_cuda_graph:
                # When mha_return_lse is True (chunked prefix MHA), the backend
                # returns a tuple (output, lse). The bridge/breakable path can't
                # propagate tuples, so call the backend directly.
                if getattr(forward_batch, "mha_return_lse", False):
                    return forward_batch.attn_backend.forward(
                        q, k, v, self, forward_batch, save_kv_cache, **kwargs
                    )

                from sglang.srt.model_executor.breakable_piecewise_cuda_graph_runner import (
                    get_bridge_buffers,
                )

                bridges = get_bridge_buffers()
                if bridges is not None and "k_rope" not in kwargs:
                    bridges.ensure_size(self)
                    bq = bridges.q.flatten()[: q.numel()].view(q.shape)
                    bk = (
                        bridges.k.flatten()[: k.numel()].view(k.shape)
                        if k is not None
                        else None
                    )
                    bv = (
                        bridges.v.flatten()[: v.numel()].view(v.shape)
                        if v is not None
                        else None
                    )
                    bq.copy_(q)
                    if bk is not None:
                        bk.copy_(k)
                    if bv is not None:
                        bv.copy_(v)
                    n = q.shape[0]
                    out_dim = self.tp_q_head_num * self.v_head_dim
                    output = bridges.output.flatten()[: n * out_dim].view(
                        n, out_dim
                    )
                    del q, k, v
                    breakable_unified_attention_with_output(
                        bq,
                        bk,
                        bv,
                        output,
                        save_kv_cache,
                        self.layer_id,
                        _attention_layer=self,
                        **kwargs,
                    )
                    return output
                else:
                    output = (
                        q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
                        if self.qk_head_dim != self.v_head_dim
                        else torch.empty_like(q)
                    )
                    breakable_unified_attention_with_output(
                        q,
                        k,
                        v,
                        output,
                        save_kv_cache,
                        self.layer_id,
                        _attention_layer=self,
                        **kwargs,
                    )
                    return output
            else:
                if self.qk_head_dim != self.v_head_dim:
                    output = q.new_empty(
                        (q.shape[0], self.tp_q_head_num * self.v_head_dim)
                    )
                else:
                    output = torch.empty_like(q)
                unified_attention_with_output(
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
                return output
        else:
            return forward_batch.attn_backend.forward(
                q,
                k,
                v,
                self,
                forward_batch,
                save_kv_cache,
                **kwargs,
            )


def _bcg_need_mha_fixup(
    forward_batch: "ForwardBatch",
    save_kv_cache: bool,
    k_rope: Optional[torch.Tensor],
    query: torch.Tensor,
) -> Optional[int]:
    """Check if this is an MHA layer in BCG replay that needs special handling.

    Returns the number of tokens to slice q/k/v to, or None if no fixup needed.
    MHA layers (save_kv_cache=False, no k_rope) receive static-sized tensors
    from graph capture but the attention backend expects real-sized tensors.
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


def _unified_attention_with_output_impl(
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
    _attention_layer: Optional["RadixAttention"] = None,
) -> None:
    """Attention implementation shared by both torch.compile split-op and breakable CUDA graph paths."""
    context = get_forward_context()
    forward_batch = context.forward_batch
    if _attention_layer is not None:
        attention_layer = _attention_layer
    else:
        attention_layers = context.attention_layers
        attention_layer = attention_layers[layer_id]

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope
    if k_rope is not None:
        kwargs["k_rope"] = k_rope
    if sinks is not None:
        kwargs["sinks"] = sinks

    _bcg_n = _bcg_need_mha_fixup(forward_batch, save_kv_cache, k_rope, query)
    if _bcg_n is not None:
        ret = _bcg_mha_attention(
            query, key, value, attention_layer, forward_batch,
            save_kv_cache, _bcg_n, kwargs,
        )
        output[:_bcg_n].view(ret.shape).copy_(ret)
    else:
        ret = forward_batch.attn_backend.forward(
            query, key, value, attention_layer, forward_batch, save_kv_cache,
            **kwargs,
        )
        assert (
            output.numel() == ret.numel()
        ), f"Output tensor element mismatch: {output.numel()} != {ret.numel()}"
        output.view(ret.shape).copy_(ret)
    return


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
    """MHA attention for BCG replay.

    During BCG replay, MHA layers receive static-sized tensors from graph
    capture but the attention backend expects real-sized tensors. This function
    slices q/k/v to extend_num_tokens, runs extend-only attention, then runs
    the chunked prefix attention loop (which doesn't replay in CUDA graphs
    since it's Python code between break points).

    The BCG runner pre-computes _bcg_mha_slice_n and _bcg_mha_has_prefix on
    forward_batch once per forward to avoid redundant per-layer evaluation.
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
        query, key, value, attention_layer, forward_batch, save_kv_cache,
        **kwargs,
    )

    # Run chunked prefix attention and merge
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

    # Restore flags
    forward_batch.set_attn_attend_prefix_cache(False)
    forward_batch.mha_return_lse = False
    forward_batch.mha_one_shot = True

    return ret


# Lazy-init: wrapped with non_graph(True) on first use to avoid import-time
# dependency on breakable_cuda_graph (which requires cuda.bindings).
_breakable_attention_fn = None


def breakable_unified_attention_with_output(
    query, key, value, output, save_kv_cache, layer_id, **kwargs
):
    global _breakable_attention_fn
    if _breakable_attention_fn is None:
        from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
            eager_on_graph,
        )

        _breakable_attention_fn = eager_on_graph(True)(_unified_attention_with_output_impl)
    return _breakable_attention_fn(
        query, key, value, output, save_kv_cache, layer_id, **kwargs
    )


@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_attention_with_output(
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
    context = get_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope
    if k_rope is not None:
        kwargs["k_rope"] = k_rope
    if sinks is not None:
        kwargs["sinks"] = sinks

    ret = forward_batch.attn_backend.forward(
        query, key, value, attention_layer, forward_batch, save_kv_cache, **kwargs
    )
    assert (
        output.numel() == ret.numel()
    ), f"Output tensor element mismatch: {output.numel()} != {ret.numel()}"

    output.view(ret.shape).copy_(ret)
    return
