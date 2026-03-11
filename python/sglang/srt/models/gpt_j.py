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

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gpt_j.py
"""Inference-only GPT-J model compatible with HuggingFace weights."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import GPTJConfig

from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import get_act_fn
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
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix


class GPTJAttention(nn.Module):

    def __init__(
        self,
        layer_id: int,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        total_num_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // total_num_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )

        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % tensor_model_parallel_world_size == 0
        num_heads = total_num_heads // tensor_model_parallel_world_size

        scaling = head_dim**-0.5
        assert getattr(config, "rotary", True)
        assert config.rotary_dim % 2 == 0
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=config.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            num_heads,
            head_dim,
            scaling=scaling,
            num_kv_heads=num_heads,
            layer_id=layer_id,
            quant_config=quant_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output, _ = self.out_proj(attn_output)
        return attn_output


class GPTJMLP(nn.Module):

    def __init__(
        self,
        intermediate_size: int,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.n_embd
        self.fc_in = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("fc_in", prefix),
        )
        self.fc_out = RowParallelLinear(
            intermediate_size,
            hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("fc_out", prefix),
        )

        self.act = get_act_fn(config.activation_function)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class GPTJBlock(nn.Module):

    def __init__(
        self,
        layer_id: int,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        inner_dim = 4 * config.n_embd if config.n_inner is None else config.n_inner
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTJAttention(
            layer_id,
            config,
            quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = GPTJMLP(
            inner_dim,
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        mlp_output = self.mlp(hidden_states)
        hidden_states = attn_output + mlp_output + residual
        return hidden_states


class GPTJModel(nn.Module):

    def __init__(
        self,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        embed_dim = config.n_embd
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            embed_dim,
        )
        self.h = nn.ModuleList(
            [
                GPTJBlock(
                    i,
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"h.{i}", prefix),
                )
                for i in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        for layer in self.h:
            hidden_states = layer(positions, hidden_states, forward_batch)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPTJForCausalLM(nn.Module):

    def __init__(
        self,
        config: GPTJConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        assert not config.tie_word_embeddings
        self.quant_config = quant_config
        self.transformer = GPTJModel(
            config,
            quant_config,
            prefix=add_prefix("transformer", prefix),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.n_embd,
            bias=True,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids, positions, forward_batch, inputs_embeds
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "attn.bias" in name or "attn.masked_bias" in name:
                continue

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = GPTJForCausalLM
