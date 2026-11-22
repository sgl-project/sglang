# Copyright 2026 SGLang Team
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

# Adapted from modeling_aion_instruct.py in
# https://huggingface.co/microsoft/Aion-1.0-Instruct-2.3B
"""Inference-only Aion-1.0-Instruct model compatible with HuggingFace weights."""

from typing import Any, Dict, Optional

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.utils import apply_qk_norm_rope, can_use_jit_qk_norm_rope
from sglang.srt.runtime_context import get_parallel


class AionInstructAttention(nn.Module):
    """GQA with per-head Q/K RMSNorm, over full context or a sliding window."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_theta: float,
        rope_scaling: Dict[str, Any],
        max_position_embeddings: int,
        rms_norm_eps: float,
        attention_bias: bool,
        sliding_window_size: int = -1,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        attn_tp_size = get_parallel().attn_tp_size
        assert num_heads % attn_tp_size == 0
        if num_kv_heads >= attn_tp_size:
            assert num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % num_kv_heads == 0
        self.num_heads = num_heads // attn_tp_size
        self.num_kv_heads = max(1, num_kv_heads // attn_tp_size)

        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            parallel_group="attn_tp",
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            parallel_group="attn_tp",
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self.alt_stream = alt_stream
        self.use_jit_qk_norm_rope = can_use_jit_qk_norm_rope(
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            rotary_emb=self.rotary_emb,
            head_dim=self.head_dim,
            dtype=torch.get_default_dtype(),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm_rope(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            rotary_emb=self.rotary_emb,
            positions=positions,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
            use_jit=self.use_jit_qk_norm_rope,
        )
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output
