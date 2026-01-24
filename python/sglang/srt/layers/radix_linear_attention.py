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

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from torch import nn

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
        # GDN KDA Shared Weights
        conv_weights: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
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
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        return forward_batch.attn_backend.forward(
            layer=self,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        )
