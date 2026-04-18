
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
"""Simplified radix attention.

This is the newly implemented radix attention which is also a simplified version.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class AttentionType(Enum):
    """Attention type.

    Use strings to stay compatible with torch.compile.
    """

    DECODER = "decoder"
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"
    ENCODER_ONLY = "encoder_only"


class RadixAttention(nn.Module):
    """Thin attention wrapper that preserves the original public contract.

    The actual attention computation still happens inside
    ``forward_batch.attn_backend.forward(...)``. This class keeps only the core
    responsibilities needed by outer SGLang code:
      1. store layer metadata
      2. reshape K/V tensors into backend-expected layouts
      3. dispatch to either the unified extend path or the normal backend path
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

        # Public fields that outside code may rely on.
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
        self.attn_type = attn_type
        self.pos_encoding_mode = pos_encoding_mode
        self.logit_capping_method = logit_capping_method
        self.xai_temperature_len = -1

        # Keep quantization-related attributes for compatibility.
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = None
        self.v_scale_float = None
        self.quant_method = None

        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is not None:
            self.quant_method.create_weights(self)

    def _reshape_kv(self, k, v, kwargs):
        """Reshape key/value tensors into the layout expected by the backend.

        Notes:
        - In some cross-layer-sharing paths K/V can be None.
        - When ``k_rope`` is present, K uses ``v_head_dim`` just like the
          original implementation.
        """
        if k is None:
            return k, v

        assert v is not None, "Expected V to be present whenever K is present."

        if "k_rope" in kwargs:
            k = k.view(-1, self.tp_k_head_num, self.v_head_dim)
            return k, v

        k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
        v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
        return k, v

    def _should_use_unified_extend_path(self, forward_batch: ForwardBatch) -> bool:
        """Return True when the piecewise extend path should be used."""
        return (
            forward_batch.forward_mode.is_extend()
            and get_forward_context() is not None
        )

    def _allocate_output(self, q: torch.Tensor) -> torch.Tensor:
        """Allocate the output tensor for the unified extend path."""
        if self.qk_head_dim != self.v_head_dim:
            return q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
        return torch.empty_like(q)

    def _forward_backend(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        **kwargs,
    ):
        """Call the backend with the stable contract expected by SGLang."""
        return forward_batch.attn_backend.forward(
            q,
            k,
            v,
            self,
            forward_batch,
            save_kv_cache,
            **kwargs,
        )

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        k, v = self._reshape_kv(k, v, kwargs)

        if self._should_use_unified_extend_path(forward_batch):
            output = self._allocate_output(q)
            unified_attention_with_output(
                q,
                k,
                v,
                output,
                save_kv_cache,
                self.layer_id,
                **kwargs,
            )
            return output

        return self._forward_backend(
            q,
            k,
            v,
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
    """Run attention through the backend and copy the result into ``output``.

    Keeping this helper preserves the original external behavior used by the
    extend-mode custom-op path while keeping the function body minimal.
    """
    context = get_forward_context()
    assert context is not None, "Forward context is required for unified attention."

    forward_batch = context.forward_batch
    attention_layer = context.attention_layers[layer_id]

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope
    if k_rope is not None:
        kwargs["k_rope"] = k_rope
    if sinks is not None:
        kwargs["sinks"] = sinks

    ret = forward_batch.attn_backend.forward(
        query,
        key,
        value,
        attention_layer,
        forward_batch,
        save_kv_cache,
        **kwargs,
    )

    assert output.numel() == ret.numel(), (
        f"Output tensor element mismatch: {output.numel()} != {ret.numel()}"
    )
    output.view(ret.shape).copy_(ret)
