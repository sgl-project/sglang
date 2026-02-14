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
"""Inference-only LoopCoder model compatible with HuggingFace weights."""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaMLP as LoopCoderMLP
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


class LoopGateProjection(nn.Module):
    """Gate projection for mixed attention in Loop 2+.

    Computes: g = sigmoid(linear(Q)) for each head independently.
    This gate determines how much to use Loop1's KV (global) vs current loop's KV (local).

    Supports tensor parallelism: each GPU handles a subset of heads.
    The weight matrix has shape [num_heads, head_dim] and is split along the head dimension.
    """

    def __init__(
        self,
        total_num_heads: int,
        head_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.total_num_heads = total_num_heads
        self.head_dim = head_dim
        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.gate_proj = ColumnParallelLinear(
            head_dim,
            self.total_num_heads,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Compute gate values from query tensor.

        Args:
            query: [num_heads, num_tokens, head_dim]
                where num_heads is the number of heads on this TP rank
                and num_tokens = batch * seq_len

        Returns:
            gate: [num_tokens, num_heads * head_dim] (flattened format matching q shape)
        """
        num_heads, num_tokens, head_dim = query.shape

        assert (
            num_heads == self.num_heads
        ), f"Expected {self.num_heads} heads, got {num_heads}"

        query_flat = query.reshape(-1, head_dim)

        gate_logits_flat, _ = self.gate_proj(query_flat)

        gate_logits = gate_logits_flat.reshape(num_heads, num_tokens, self.num_heads)

        # Extract diagonal: each head h's query should use output column h
        gate_logits = torch.diagonal(gate_logits, dim1=0, dim2=2)
        gate_logits = gate_logits.transpose(0, 1)
        gate_logits = gate_logits.unsqueeze(-1)

        # Apply sigmoid
        gate = torch.sigmoid(gate_logits)

        # Expand and reshape to match q shape: [num_tokens, num_heads * head_dim]
        gate = gate.transpose(0, 1)
        gate = gate.expand(-1, -1, head_dim)
        gate = gate.reshape(num_tokens, num_heads * head_dim)

        return gate


class LoopCoderAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Get loop_num from config, default to 2 if not specified
        self.loop_num = getattr(config, "loop_num", 2)
        self.loop_window_size = getattr(config, "loop_window_size", 64)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(
            config, "max_position_embeddings", max_position
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # Create attention instances for each loop
        # Loop 0: global attention without sliding window for full context
        # Loop 1+: local attention with sliding window for recent tokens
        # Each loop needs a unique layer_id to avoid KV cache conflicts
        self.attn = nn.ModuleList()
        total_layers = getattr(config, "num_hidden_layers", 24)
        for loop_idx in range(self.loop_num):
            sliding_window = -1 if loop_idx == 0 else self.loop_window_size
            # Use unique layer_id for each loop: loop_idx * total_layers + layer_id
            # This ensures each loop has its own KV cache space
            unique_layer_id = loop_idx * total_layers + layer_id

            self.attn.append(
                RadixAttention(
                    self.num_heads,
                    self.head_dim,
                    self.scaling,
                    num_kv_heads=self.num_kv_heads,
                    layer_id=unique_layer_id,  # Unique layer_id for each loop
                    sliding_window_size=sliding_window,
                    quant_config=quant_config,
                    prefix=add_prefix(f"attn.{loop_idx}", prefix),
                )
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        loop_idx: int,
        gate_proj: Optional[LoopGateProjection] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        if loop_idx == 0:
            # First loop: standard global attention, save KV to cache
            attn_output = self.attn[0](q, k, v, forward_batch)
        else:
            # Loop 2+: mixed attention with learned gating
            # Global attention: read from Loop 0's KV cache without updating (save_kv_cache=False)
            # This provides full context information
            # Pass k=None, v=None to read from KV cache instead of recomputing
            global_attn_output = self.attn[0](
                q, None, None, forward_batch, save_kv_cache=False
            )

            # Local attention: use current loop's KV with sliding window
            # This focuses on recent tokens within the window
            local_attn_output = self.attn[loop_idx](q, k, v, forward_batch)

            # Compute gating weights using query-dependent projection
            assert gate_proj is not None, "gate_proj must be provided for loop_idx > 0"
            num_tokens = q.shape[0]
            q_reshaped = q.view(num_tokens, self.num_heads, self.head_dim).transpose(
                0, 1
            )
            gate = gate_proj(q_reshaped)

            # Mix global and local attention outputs with learned gate
            # gate controls the balance between global context and local focus
            attn_output = global_attn_output * gate + local_attn_output * (1 - gate)

        output, _ = self.o_proj(attn_output)
        return output


class LoopCoderDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        self.self_attn = LoopCoderAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            max_position=getattr(config, "max_position_embeddings", 4096 * 32),
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = LoopCoderMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        loop_idx: int,
        gate_proj: Optional[LoopGateProjection] = None,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            loop_idx=loop_idx,
            gate_proj=gate_proj,
        )
        hidden_states = hidden_states + residual

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class IQuestLoopCoderModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.loop_num = getattr(self.config, "loop_num", 2)
        self.window_size = getattr(self.config, "loop_window_size", 64)

        # Gate projections for Loop 2+ (one per layer)
        head_dim = config.hidden_size // config.num_attention_heads
        gate_projections = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LoopGateProjection(
                total_num_heads=config.num_attention_heads,
                head_dim=head_dim,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("gate_projections", prefix),
        )
        if isinstance(gate_projections, tuple):
            self.start_layer, self.end_layer, self.gate_projections = gate_projections
        else:
            self.start_layer, self.end_layer = 0, config.num_hidden_layers
            self.gate_projections = gate_projections

        layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LoopCoderDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        if isinstance(layers, tuple):
            self.start_layer, self.end_layer, self.layers = layers
        else:
            self.start_layer, self.end_layer = 0, config.num_hidden_layers
            self.layers = layers

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        # Multi-loop forward pass
        for loop_idx in range(self.loop_num):
            for layer_idx in range(self.start_layer, self.end_layer):
                layer = self.layers[layer_idx]
                # Get gate_proj for this layer (only for loop_idx > 0)
                gate_proj = self.gate_projections[layer_idx] if loop_idx > 0 else None
                hidden_states = layer(
                    positions, hidden_states, forward_batch, loop_idx, gate_proj
                )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class IQuestLoopCoderForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.model = IQuestLoopCoderModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Handle gate_projections weights
            if name.startswith("gate_projections."):
                if name.endswith(".weight"):
                    sglang_name = name.replace(".weight", ".gate_proj.weight")
                elif name.endswith(".bias"):
                    sglang_name = name.replace(".bias", ".gate_proj.bias")
                else:
                    continue

                if sglang_name in params_dict:
                    param = params_dict[sglang_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                continue

            # Handle stacked parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle regular parameters
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


# Entry class for model registration
EntryClass = IQuestLoopCoderForCausalLM
