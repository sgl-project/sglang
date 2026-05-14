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
from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
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
        # Set to True at model load by ModelRunner when --enable-double-sparsity
        # is set (see ModelRunner._init_double_sparsity_coordinator). Read at
        # trace time as a Python-static attribute — never branched on a device
        # tensor under graph replay.
        self.ds_enabled = False

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

        # Common path FIRST: byte-for-byte unchanged from the dense codepath
        # when DS is off (the vast majority of users).
        if not self.ds_enabled:
            return self._forward_inner(q, k, v, forward_batch, save_kv_cache, **kwargs)

        # DS path. Existing inner branches preserved; DS hooks wrap them.
        from sglang.srt.mem_cache.sparsity import get_sparse_coordinator

        coordinator = get_sparse_coordinator()
        if coordinator is None:
            # Defensive: if the coordinator was torn down but ds_enabled wasn't
            # cleared, fall back to dense rather than crash.
            return self._forward_inner(q, k, v, forward_batch, save_kv_cache, **kwargs)

        # Decode-only: try the native sparse-decode path first (bypasses FA3
        # entirely; runs its own Triton score + topk + sparse-attn kernels).
        # If it returns a tensor, the attention output is already computed —
        # skip both attention_begin (no FA3 metadata rewrite) and
        # _forward_inner (no FA3 backend call). Only attention_end runs to
        # write K_label for the new decode token.
        # Falls through to the legacy FA3 + DSFlashAttentionAdaptor path
        # when not eligible (short seq_len, missing scratch, etc.).
        if forward_batch.forward_mode.is_decode_or_idle():
            try_native = getattr(
                coordinator.algorithm, "try_native_sparse_decode", None
            )
            # The native sparse-decode path doesn't yet support q_rope /
            # k_rope / sinks (split-RoPE and sink-token attention). Other
            # kwargs (e.g. instrumentation tags) pass through harmlessly.
            _NATIVE_UNSUPPORTED_KWARGS = {"q_rope", "k_rope", "sinks"}
            if try_native is not None and not (
                kwargs.keys() & _NATIVE_UNSUPPORTED_KWARGS
            ):
                native_out = try_native(
                    q, k, v, self, forward_batch, save_kv_cache=save_kv_cache
                )
                if native_out is not None:
                    if save_kv_cache:
                        coordinator.attention_end(native_out, self, forward_batch)
                    return native_out

            attn_metadata = getattr(
                forward_batch.attn_backend, "forward_metadata", None
            )
            coordinator.attention_begin(q, k, v, self, forward_batch, attn_metadata)

        ret = self._forward_inner(q, k, v, forward_batch, save_kv_cache, **kwargs)

        # K_label write hook. Skip entirely when save_kv_cache=False —
        # otherwise the side cache desyncs from the KV pool (the inner call
        # didn't write K, so we'd read stale rows from out_cache_loc and
        # update K_label with garbage). The algorithm has a defensive
        # `getattr(forward_batch, "save_kv_cache", True)` check too, but
        # save_kv_cache lives on this function arg, not on forward_batch.
        if save_kv_cache:
            coordinator.attention_end(ret, self, forward_batch)
        return ret

    def _forward_inner(self, q, k, v, forward_batch, save_kv_cache, **kwargs):
        """Existing inner attention dispatch — DO NOT MODIFY semantics here.

        Two paths preserved verbatim from main:
          - extend with `get_forward_context()`: through the registered split
            op `unified_attention_with_output` (or the bcg variant).
          - everything else: direct `attn_backend.forward(...)`.
        """
        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
            if self.qk_head_dim != self.v_head_dim:
                output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
            else:
                output = torch.empty_like(q)
            if is_in_breakable_cuda_graph():
                bcg_unified_attention_with_output(
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            else:
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
    real_num_tokens = forward_batch.num_token_non_padded_cpu

    query = query[:real_num_tokens]
    key = key[:real_num_tokens]
    value = value[:real_num_tokens]

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope[:real_num_tokens]
    if k_rope is not None:
        kwargs["k_rope"] = k_rope[:real_num_tokens]
    if sinks is not None:
        kwargs["sinks"] = sinks

    original_out_cache_loc = forward_batch.out_cache_loc
    original_out_cache_loc_swa = forward_batch.out_cache_loc_swa
    token_to_kv_pool = forward_batch.token_to_kv_pool
    original_swa_loc = getattr(token_to_kv_pool, "swa_loc", None)
    # Keep the original ForwardBatch object and only narrow cache locations for
    # this backend call so model/backend state is still written to the same batch.
    forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]
    if original_out_cache_loc_swa is not None:
        forward_batch.out_cache_loc_swa = original_out_cache_loc_swa[:real_num_tokens]
        if hasattr(token_to_kv_pool, "set_swa_loc"):
            token_to_kv_pool.set_swa_loc(forward_batch.out_cache_loc_swa)

    # Store pre-allocated output for FA backend to write directly into.
    # Must slice to real_num_tokens to match the narrowed query shape —
    # the FA kernel validates out.size(0) == q.size(0).
    forward_batch._attn_output = output[:real_num_tokens]

    ret = forward_batch.attn_backend.forward(
        query,
        key,
        value,
        attention_layer,
        forward_batch,
        save_kv_cache,
        **kwargs,
    )
    forward_batch.out_cache_loc = original_out_cache_loc
    forward_batch.out_cache_loc_swa = original_out_cache_loc_swa
    if original_out_cache_loc_swa is not None and hasattr(
        token_to_kv_pool, "set_swa_loc"
    ):
        token_to_kv_pool.set_swa_loc(original_swa_loc)

    if ret.data_ptr() != output.data_ptr():
        output[:real_num_tokens].view(ret.shape).copy_(ret)
    return


bcg_unified_attention_with_output = eager_on_graph(True)(unified_attention_with_output)
