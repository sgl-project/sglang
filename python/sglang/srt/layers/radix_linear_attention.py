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
from typing import Optional

import torch
from torch import nn

from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import direct_register_custom_op


class RadixLinearAttention(nn.Module):
    """
    The Linear Attention Layer Implementation.
    """

    def __init__(
        self,
        layer_id: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        head_v_dim: int,
        # GDN Specific Weights
        conv_weights: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.head_v_dim = head_v_dim
        self.q_per_kv = num_q_heads // num_kv_heads

        self.conv_weights = conv_weights
        self.bias = bias
        self.activation = activation
        self.A_log = A_log
        self.dt_bias = dt_bias

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
            output = torch.empty(
                mixed_qkv.shape[0],
                self.num_q_heads * self.head_v_dim,
                device=mixed_qkv.device,
                dtype=mixed_qkv.dtype,
            )
            torch.ops.sglang.unified_linear_attention_with_output(
                mixed_qkv, output, save_kv_cache, self.layer_id, a, b
            )
            return output
        else:
            return forward_batch.attn_backend.forward(
                mixed_qkv=mixed_qkv,
                layer=self,
                a=a,
                b=b,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
            )


def unified_linear_attention_with_output(
    mixed_qkv: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> None:
    context = get_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]

    ret = forward_batch.attn_backend.forward(
        mixed_qkv=mixed_qkv,
        layer=attention_layer,
        a=a,
        b=b,
        forward_batch=forward_batch,
        save_kv_cache=save_kv_cache,
    )

    assert output.shape == ret.shape
    output.copy_(ret)
    return


def unified_linear_attention_with_output_fake(
    mixed_qkv: torch.Tensor,
    output: torch.Tensor,
    save_kv_cache: bool,
    layer_id: int,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_linear_attention_with_output",
    op_func=unified_linear_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_linear_attention_with_output_fake,
)
