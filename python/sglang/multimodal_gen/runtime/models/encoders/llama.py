# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/llama.py

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

# from ..utils import (extract_layer_index)
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput, LlamaConfig
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul

# from vllm.model_executor.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.rotary_embedding import get_rope
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            # output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tp_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        # Phi models introduced a partial_rotary_factor parameter in the config
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        is_gguf = (
            quant_config
            and hasattr(quant_config, "get_name")
            and quant_config.get_name() == "gguf"
        )
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

        self.attn = LocalAttention(
            self.num_heads,
            self.head_dim,
            self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=True,
            supported_attention_backends=config._supported_attention_backends,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        # attn_output = self.attn(q, k, v)
        # use flash_attn_func
        # TODO (Attn abstraction and backend)
        # reshape q, k, v to (batch_size, seq_len, num_heads, head_dim)
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        # import pdb; pdb.set_trace()
        # attn_output = flash_attn_varlen_func(q, k, v, softmax_scale=self.scaling, causal=True)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(TextEncoder):

    def __init__(
        self,
        config: LlamaConfig,
    ):
        super().__init__(config)

        self.config = config
        self.quant_config = self.config.quant_config
        if config.lora_config is not None:
            max_loras = 1
            lora_vocab_size = 1
            if hasattr(config.lora_config, "max_loras"):
                max_loras = config.lora_config.max_loras
            if hasattr(config.lora_config, "lora_extra_vocab_size"):
                lora_vocab_size = config.lora_config.lora_extra_vocab_size
            lora_vocab = lora_vocab_size * max_loras
        else:
            lora_vocab = 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=config.quant_config,
        )

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config=config,
                    quant_config=config.quant_config,
                    prefix=f"{config.prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None

        if position_ids is None:
            position_ids = torch.arange(
                0, hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)

        all_hidden_states: tuple[Any, ...] | None = () if output_hidden_states else None
        for layer in self.layers:
            if all_hidden_states is not None:
                # TODO
                all_hidden_states += (
                    (hidden_states,)
                    if residual is None
                    else (hidden_states + residual,)
                )
            hidden_states, residual = layer(position_ids, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)

        # add hidden states from the last decoder layer
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        # TODO(will): maybe unify the output format with other models and use
        # our own class
        output = BaseEncoderOutput(
            last_hidden_state=hidden_states,
            # past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            # attentions=all_self_attns,
        )

        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # if (self.quant_config is not None and
            #     (scale_name := self.quant_config.get_cache_scale(name))):
            #     # Loading kv cache quantization scales
            #     param = params_dict[scale_name]
            #     weight_loader = getattr(param, "weight_loader",
            #                             default_weight_loader)
            #     loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
            #                      loaded_weight[0])
            #     weight_loader(param, loaded_weight)
            #     loaded_params.add(scale_name)
            #     continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                kv_scale_name: str | None = maybe_remap_kv_scale_name(name, params_dict)
                if kv_scale_name is None:
                    continue
                else:
                    name = kv_scale_name
            for (
                param_name,
                weight_name,
                shard_id,
            ) in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


EntryClass = LlamaModel
