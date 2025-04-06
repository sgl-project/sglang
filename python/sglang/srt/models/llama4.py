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
# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama4.py
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

import torch
from torch import nn
from transformers import Llama4TextConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM, LlamaMLP, LlamaModel
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


class Llama4MoE(nn.Module):

    @staticmethod
    def custom_routing_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        router_scores_aK, router_indices_aK = torch.topk(gating_output, topk, dim=-1)
        router_scores_aK = torch.sigmoid(router_scores_aK.float()).to(
            hidden_states.dtype
        )
        return (
            router_scores_aK.view(-1).reshape(router_scores_aK.shape),
            router_indices_aK.to(torch.int32),
        )

    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok

        intermediate_size_moe = config.intermediate_size
        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("router", prefix),
        )

        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            custom_routing_function=Llama4MoE.custom_routing_function,
            intermediate_size=intermediate_size_moe,
            reduce_results=False,
            renormalize=False,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

        self.shared_expert = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            hidden_act="silu",
            quant_config=quant_config,
            prefix=add_prefix("shared_expert", prefix),
            reduce_results=False,  # We need to do scatter before reduce
        )

    def forward(self, hidden_states):
        # router_scores: [num_tokens, num_experts]
        router_logits, _ = self.router(hidden_states)
        shared_out = self.shared_expert(hidden_states)
        routed_out = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        out_aD = routed_out + shared_out

        if self.tp_size > 1:
            out_aD = tensor_model_parallel_all_reduce(out_aD)

        return out_aD


class Llama4Attention(nn.Module):

    def __init__(
        self,
        config: Llama4TextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.no_rope_layers = config.no_rope_layers
        self.nope = self.no_rope_layers[self.layer_id] == 0
        self.use_qk_norm = config.use_qk_norm and not self.nope
        tp_size = get_tensor_model_parallel_world_size()
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
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_temperature_tuning = (
            getattr(config, "attn_temperature_tuning", False) and self.nope
        )
        self.floor_scale = getattr(config, "floor_scale", 8192.0)
        self.attn_scale = getattr(config, "attn_scale", 0.1)
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.n_rep = self.num_heads // self.num_kv_heads
        self.q_norm = (
            RMSNorm(
                hidden_size=self.q_size,
                eps=config.rms_norm_eps,
            )
            if self.use_qk_norm
            else None
        )
        self.k_norm = (
            RMSNorm(
                hidden_size=self.kv_size,
                eps=config.rms_norm_eps,
            )
            if self.use_qk_norm
            else None
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=int(rope_theta),
                rope_scaling=rope_scaling if rope_scaling != "default" else None,
                is_neox_style=is_neox_style,
            )
            if not self.nope
            else None
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def _get_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        floor = torch.floor((positions + 1.0) / self.floor_scale)
        attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0

        return attn_scale.unsqueeze(-1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399) to NoPE layers, where
        # the inference-time temperature tuning function is customized to not affect short context
        # while working at very long context
        # https://arxiv.org/abs/2501.19399
        if self.attn_temperature_tuning and self.nope:
            attn_scale = self._get_attn_scale(positions)
            q = (q * attn_scale).to(q.dtype)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        if self.q_norm is not None:
            # TODO: support float
            q = self.q_norm(q.bfloat16().contiguous()).to(q.dtype)
        if self.k_norm is not None:
            k = self.k_norm(k.bfloat16().contiguous()).to(k.dtype)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Llama4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_theta
        rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings

        self.self_attn = Llama4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            bias_o_proj=False,
            prefix=add_prefix("self_attn", prefix),
        )
        is_moe_layer = (layer_id + 1) % config.interleave_moe_layer_step == 0
        if is_moe_layer:
            self.feed_forward = Llama4MoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
        else:
            self.feed_forward = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size_mlp,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
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
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class Llama4Model(LlamaModel):
    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.num_experts = getattr(config, "num_local_experts", 0)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Llama4DecoderLayer(
                config=config, layer_id=idx, quant_config=quant_config, prefix=prefix
            ),
            prefix="model.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []

    def load_moe_expert_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: Dict[str, nn.Parameter],
        loaded_params: Set[str],
        expert_params_mapping: List[Tuple[str, str, int, str]],
        fused: bool = True,
    ) -> bool:
        expert_param_loaded = False
        if "experts.gate_up_proj" in name:
            loaded_weight = loaded_weight.chunk(2, dim=-1)
        for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
            new_loaded_weight = loaded_weight
            if fused:
                e_str, _, proj_str, _ = weight_name.split(".")
                weight_name = f"{e_str}.{proj_str}"
                param_name = f"{param_name}weight"
            if weight_name not in name:
                continue
            full_param_name = name.replace(weight_name, param_name)
            if full_param_name not in params_dict:
                continue
            param = params_dict[full_param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if fused:
                if "w13" in full_param_name:
                    shard_idx = 0 if shard_id == "w1" else 1
                    new_loaded_weight = new_loaded_weight[shard_idx]
                new_loaded_weight = new_loaded_weight.transpose(-1, -2)
                layer_idx = int(name.split(".")[2])
                # EP mapping
                expert_map = self.layers[layer_idx].feed_forward.experts.expert_map
                if expert_map is not None:
                    local_expert_indices = (
                        (expert_map != -1)
                        .nonzero()
                        .flatten()
                        .to(new_loaded_weight.device)
                    )
                    new_loaded_weight = new_loaded_weight[local_expert_indices]
                    expert_id = local_expert_indices[0].item()
            else:
                # TODO: add EP support for non fused weights
                pass
            weight_loader(
                param,
                new_loaded_weight,
                full_param_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )

            loaded_params.add(full_param_name)
            expert_param_loaded = True
        return expert_param_loaded


class Llama4ForCausalLM(LlamaForCausalLM):

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, prefix)

    def _init_model(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return Llama4Model(config, quant_config=quant_config, prefix=prefix)

    def permute_qk_weight_for_rotary(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        modules = name.split(".")

        # rotary embeds should be sliced
        if ("wk" in modules or "k_proj" in modules) and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight, self.config.num_key_value_heads)
        elif ("wq" in modules or "q_proj" in modules) and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight, self.config.num_attention_heads)

        return name, loaded_weight


EntryClass = [Llama4ForCausalLM]
