# Copyright 2025-2026 SGLang Team
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
"""Radix linear attention."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class RadixLinearAttention(nn.Module):
    """
    The Linear Attention Layer Implementation.
    """

    def __init__(
        self,
        layer_id: int,
        num_qk_heads: int,
        num_v_heads: int,
        head_qk_dim: int,
        head_v_dim: int,
        attention_tp_size: int = 1,
        conv_weights: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        # Q and K share the same head count and dimension (per-TP values)
        self.num_qk_heads = num_qk_heads
        self.num_v_heads = num_v_heads
        self.head_qk_dim = head_qk_dim
        self.head_v_dim = head_v_dim
        self.attention_tp_size = attention_tp_size

        self.qk_dim_per_tp = num_qk_heads * head_qk_dim
        self.value_dim_per_tp = num_v_heads * head_v_dim

        self.key_dim = self.qk_dim_per_tp * attention_tp_size
        self.value_dim = self.value_dim_per_tp * attention_tp_size

        self.num_k_heads = num_qk_heads
        self.num_q_heads = num_qk_heads
        self.head_k_dim = head_qk_dim

        self.conv_weights = conv_weights
        self.bias = bias
        self.activation = activation
        self.A_log = A_log
        self.dt_bias = dt_bias

    def forward(
        self,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        # mixed_qkv is expected in kwargs, matching the backend interface
        mixed_qkv = kwargs["mixed_qkv"]

        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
            # Output dimension is per-TP value dimension
            output = torch.empty(
                mixed_qkv.shape[0],
                self.value_dim_per_tp,
                device=mixed_qkv.device,
                dtype=mixed_qkv.dtype,
            )
            torch.ops.sglang.unified_linear_attention_with_output(
                output, self.layer_id, **kwargs
            )
            return output
        else:
            # Backend extracts layer params from self (layer_id, A_log, dt_bias, etc.)
            return forward_batch.attn_backend.forward(
                q=None,
                k=None,
                v=None,
                layer=self,
                forward_batch=forward_batch,
                **kwargs,
            )


@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_linear_attention_with_output(
    output: torch.Tensor,
    layer_id: int,
    *,
    # Runtime tensors - must be passed
    mixed_qkv: Optional[torch.Tensor] = None,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
) -> None:
    context = get_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]

    kwargs = {}
    if mixed_qkv is not None:
        kwargs["mixed_qkv"] = mixed_qkv
    if a is not None:
        kwargs["a"] = a
    if b is not None:
        kwargs["b"] = b
    if z is not None:
        kwargs["z"] = z

    ret = forward_batch.attn_backend.forward(
        q=None,
        k=None,
        v=None,
        layer=attention_layer,
        forward_batch=forward_batch,
        **kwargs,
    )

    assert (
        output.numel() == ret.numel()
    ), f"Output tensor element mismatch: {output.numel()} != {ret.numel()}"

    output.view(ret.shape).copy_(ret)
    return
