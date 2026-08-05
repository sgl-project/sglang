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

"""Inference-only GPT-NeoX model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import GPTNeoXConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers


class GPTNeoXAttention(nn.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads
        self.bias = getattr(config, "attention_bias", True)
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size

        self.query_key_value = QKVParallelLinear(
            config.hidden_size,
            self.head_size,
            self.total_num_heads,
            bias=self.bias,
            quant_config=quant_config,
            prefix=add_prefix("query_key_value", prefix),
        )

        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=self.bias,
            quant_config=quant_config,
            prefix=add_prefix("dense", prefix),
        )

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rotary_emb = get_rope(
            self.head_size,
            rotary_dim=self.head_size,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
        )

        scaling = self.head_size**-0.5
        self.attn = RadixAttention(
            self.num_heads,
            self.head_size,
            scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.dense(attn_output)
        return output


class GPTNeoXMLP(nn.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("dense_h_to_4h", prefix),
        )
        self.dense_4h_to_h = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("dense_4h_to_h", prefix),
        )
        self.act = get_act_fn(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states, _ = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.dense_4h_to_h(hidden_states)
        return hidden_states


class GPTNeoXLayer(nn.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = GPTNeoXAttention(
            config, layer_id, quant_config, prefix=add_prefix("attention", prefix)
        )
        self.mlp = GPTNeoXMLP(config, quant_config, prefix=add_prefix("mlp", prefix))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states

        attn_input = self.input_layernorm(hidden_states)
        attn_output = self.attention(
            positions=positions,
            hidden_states=attn_input,
            forward_batch=forward_batch,
        )

        if self.use_parallel_residual:
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output + attn_output
        else:
            attn_output = attn_output + hidden_states
            residual = attn_output
            mlp_input = self.post_attention_layernorm(attn_output)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_output

        return hidden_states, residual


class GPTNeoXModel(nn.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_in = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("embed_in", prefix),
            )
        else:
            self.embed_in = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: GPTNeoXLayer(
                config=config, layer_id=idx, quant_config=quant_config, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="layers",
            return_tuple=True,
        )

        if self.pp_group.is_last_rank:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
        else:
            self.final_layer_norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is not None:
                hidden_states = input_embeds
            else:
                hidden_states = self.embed_in(input_ids)
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if residual is not None:
            hidden_states = hidden_states + residual

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class GPTNeoXForCausalLM(nn.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.gpt_neox = GPTNeoXModel(
            config, quant_config, prefix=add_prefix("gpt_neox", prefix)
        )
        self.embed_out = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_out", prefix),
        )

        if self.config.tie_word_embeddings:
            self.embed_out.weight = self.gpt_neox.embed_in.weight

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.pp_group = get_pp_group()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        hidden_states = self.gpt_neox(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids, hidden_states, self.embed_out, forward_batch
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.gpt_neox.start_layer
                or layer_id >= self.gpt_neox.end_layer
            ):
                continue

            skip_keys = [
                "attention.bias",
                "attention.masked_bias",
                "rotary_emb.inv_freq",
                "rotary_emb.cos_cached",
                "rotary_emb.sin_cached",
            ]
            if any(key in name for key in skip_keys) or name not in params_dict:
                continue

            param = params_dict[name]

            if "query_key_value" in name:
                output_dim = getattr(param, "output_dim", None)
                num_heads = self.config.num_attention_heads

                if output_dim is not None:
                    loaded_weight_shape = loaded_weight.shape
                    loaded_weight = loaded_weight.view(
                        loaded_weight_shape[:output_dim]
                        + (num_heads, 3, -1)
                        + loaded_weight_shape[output_dim + 1 :]
                    )
                    loaded_weight = loaded_weight.transpose(output_dim, output_dim + 1)
                    loaded_weight = loaded_weight.reshape(loaded_weight_shape)

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = GPTNeoXForCausalLM
