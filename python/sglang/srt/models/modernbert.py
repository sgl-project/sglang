# Copyright 2023-2025 SGLang Team
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

"""ModernBert encoder-only model for embedding use cases."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import ModernBertConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix


class ModernBertEmbeddings(nn.Module):
    """Token embedding + layer norm + dropout."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=config.norm_bias,
        )
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.tok_embeddings(input_ids)
        hidden_states = self.drop(self.norm(hidden_states))
        return hidden_states


class ModernBertSelfAttention(nn.Module):
    """RadixAttention based self-attention for ModernBERT."""

    def __init__(
        self,
        config: ModernBertConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads",
            )

        self.layer_id = layer_id
        global_attn_every_n_layers = getattr(config, "global_attn_every_n_layers", 0)
        self.global_attn_interval = max(
            global_attn_every_n_layers,
            1,
        )
        self.local_attention = getattr(config, "local_attention", -1)
        self.is_global_layer = (
            global_attn_every_n_layers <= 0 or layer_id % self.global_attn_interval == 0
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size
        self.total_num_kv_heads = self.total_num_heads
        assert self.total_num_kv_heads % tp_world_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_world_size

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )

        if self.is_global_layer:
            rope_theta = config.global_rope_theta
            max_position_embeddings = config.max_position_embeddings
        else:
            rope_theta = (
                config.local_rope_theta
                if config.local_rope_theta is not None
                else config.global_rope_theta
            )
            max_position_embeddings = config.local_attention

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=None,
            is_neox_style=True,
            dtype=torch.float16,
        )
        sliding_window_size = -1
        if not self.is_global_layer and self.local_attention > 0:
            # The sliding window is symmetric.
            single_side = self.local_attention // 2
            if single_side > 0:
                sliding_window_size = single_side

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            attn_type=AttentionType.ENCODER_ONLY,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Apply rotary embeddings
        positions = forward_batch.positions
        max_pos = self.rotary_emb.max_position_embeddings - 1
        positions = torch.clamp(positions, 0, max_pos)
        if q.dtype == torch.float32:
            # For float32, use forward_native to avoid precision loss
            q, k = self.rotary_emb.forward_native(positions, q, k)
        else:
            q, k = self.rotary_emb.forward(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class ModernBertMLP(nn.Module):
    """ModernBERT MLP uses gated activation."""

    def __init__(
        self,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.gate_up_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size * 2,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        hidden_act = getattr(config, "hidden_activation", "gelu")
        self.activation = get_act_fn(hidden_act)
        self.dropout = nn.Dropout(config.mlp_dropout)
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        input, gate = gate_up.chunk(2, dim=-1)
        hidden_states = self.activation(input) * gate
        hidden_states = self.dropout(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class ModernBertEncoderLayer(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id

        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(
                config.hidden_size,
                eps=config.norm_eps,
                bias=config.norm_bias,
            )

        self.attention = ModernBertSelfAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attention", prefix),
        )
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=config.norm_bias,
        )
        self.mlp = ModernBertMLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attention(hidden_states, forward_batch)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ModernBertEncoder(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ModernBertEncoderLayer(
                    config=config,
                    layer_id=layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=config.norm_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, forward_batch)
        hidden_states = self.final_norm(hidden_states)
        return hidden_states


class ModernBertModel(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.encoder = ModernBertEncoder(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert get_embedding, "ModernBertModel is only used for embedding"

        # 1. Embeddings
        if input_embeds is None:
            hidden_states = self.embeddings(input_ids=input_ids)
        else:
            hidden_states = self.embeddings(inputs_embeds=input_embeds)

        # 2. Encoder
        hidden_states = self.encoder(hidden_states, forward_batch)

        # 3. Pooling
        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        replace_rules = (
            ("model.embeddings.tok_embeddings", "embeddings.tok_embeddings"),
            ("model.embeddings.norm", "embeddings.norm"),
            ("model.layers.", "encoder.layers."),
            ("model.final_norm", "encoder.final_norm"),
            (".attn.Wqkv", ".attention.qkv_proj"),
            (".attn.Wo", ".attention.out_proj"),
            (".mlp.Wi", ".mlp.gate_up_proj"),
            (".mlp.Wo", ".mlp.down_proj"),
        )

        for name, loaded_weight in weights:
            for src, dst in replace_rules:
                if src in name:
                    name = name.replace(src, dst)

            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(
                param,
                "weight_loader",
                default_weight_loader,
            )
            weight_loader(param, loaded_weight)


class ModernBertForMaskedLM(ModernBertModel):
    """ModernBertForMaskedLM is not implemented in SGLang. "
    Please use ModernBertModel for embedding tasks only.
    """

    pass


EntryClass = [ModernBertModel, ModernBertForMaskedLM]
