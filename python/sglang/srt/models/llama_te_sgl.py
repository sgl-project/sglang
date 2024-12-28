"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
# zhuohaol: Adapted from:
# https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html
"""Inference-only LLaMA model compatible with HuggingFace weights."""

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import transformer_engine as te
from torch import nn
from transformers import LlamaConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention

# from sglang.srt.layers.torchao_utils import apply_torchao_config_
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# from transformer_engine.pytorch.attention import RotaryPositionEmbedding
# from transformer_engine.pytorch.fp8 import fp8_model_init


class TELlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
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
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class TELlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = TELlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layernorm_mlp = te.pytorch.LayerNormMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            eps=config.rms_norm_eps,
            tp_size=tp_size,
            bias=False,
            return_layernorm_output=True,
            return_layernorm_output_gathered=True,
            set_parallel_mode=True,
            ub_bulk_wgrad=True,
            ub_bulk_dgrad=True,
            ub_overlap_rs_dgrad=True,
            ub_overlap_rs=True,
            ub_overlap_ag=True,
            normalization="RMSNorm",
            activation="swiglu",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        # Fully Connected with TE
        hidden_states, residual = self.layernorm_mlp(
            hidden_states
        )  # set return_layernorm_output = true
        return hidden_states, residual


class TELlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # embedding layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # multi-layer transformer decoder = attention + MLP
        self.layers = nn.ModuleList(
            [
                TELlamaDecoderLayer(
                    config, i, quant_config=quant_config, prefix=f"model.layers.{i}"
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# the complete tellama model
class TELlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.torchao_config = global_server_args_dict["torchao_config"]

        # backbone of llama model
        self.model = TELlamaModel(config, quant_config=quant_config)
        # head
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # logits processing
        self.logits_processor = LogitsProcessor(config)

        print(
            "sucessfully load TE llama"
        )  # zhuohaol: test if TELlamaForCausalLM imported

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> LogitsProcessorOutput:
        # generate hidden states
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        # use LM head mapping hidden states to logits
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head.weight, forward_batch
        )

    def get_hidden_dim(self, module_name):
        # return input_dim, output_dim
        if module_name in ["q_proj", "o_proj", "qkv_proj"]:
            return self.config.hidden_size, self.config.hidden_size
        elif module_name in ["kv_proj"]:
            return self.config.hidden_size, self.config.hidden_size // (
                self.config.num_attention_heads // self.config.num_key_value_heads
            )
        elif module_name == "gate_up_proj":
            return self.config.hidden_size, self.config.intermediate_size
        elif module_name == "down_proj":
            return self.config.intermediate_size, self.config.hidden_size
        else:
            raise NotImplementedError()

    def get_module_name(self, name):
        params_mapping = {
            "q_proj": "qkv_proj",
            "k_proj": "qkv_proj",
            "v_proj": "qkv_proj",
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
        }
        return params_mapping.get(name, name)

    def get_module_name_from_weight_name(self, name):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id, num_shard)
            ("qkv_proj", "q_proj", "q", 3),
            ("qkv_proj", "k_proj", "k", 3),
            ("qkv_proj", "v_proj", "v", 3),
            ("gate_up_proj", "gate_proj", 0, 2),
            ("gate_up_proj", "up_proj", 1, 2),
            # TODO: (zhuohaol) need to be updated later for get_module_name usage in other files
        ]
        for param_name, weight_name, shard_id, num_shard in stacked_params_mapping:
            if weight_name in name:
                return (
                    name.replace(weight_name, param_name)[: -len(".weight")],
                    num_shard,
                )
        return name[: -len(".weight")], 1

    def get_num_params(self):
        params_dict = dict(self.named_parameters())
        return len(params_dict)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # define the mapping relationship of parameters, used to handle stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (
                ".qkv_proj",
                ".q_proj.weight",
                "q",
            ),  # map q_proj to the first part of qkv_proj
            (
                ".qkv_proj",
                ".k_proj.weight",
                "k",
            ),  # map k_proj to the second part of qkv_proj
            (
                ".qkv_proj",
                ".v_proj.weight",
                "v",
            ),  # map v_proj to the third part of qkv_proj
            # map MLP layer parameters
            (
                ".layernorm_mlp.fc1_weight",
                ".gate_proj.weight",
                0,
            ),  # map gate_proj to the first part of fc1
            (
                ".layernorm_mlp.fc1_weight",
                ".up_proj.weight",
                1,
            ),  # map up_proj to the second part of fc1
            (
                ".layernorm_mlp.fc2_weight",
                ".down_proj.weight",
                0,
            ),  # map down_proj to the first part of fc2
            # map LayerNorm parameters
            (".layernorm_mlp.layer_norm_weight", ".post_attention_layernorm.weight", 0),
        ]

        # get the dictionary of all parameters of the model
        params_dict = dict(self.named_parameters())

        # traverse weights to load
        for name, loaded_weight in weights:
            # skip unnecessary weights
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            # if the weight is empty, skip
            if loaded_weight.numel() == 0:
                print(f"Warning: loaded_weight for {name} is empty.")
                continue

            # traverse the mapping relationship to handle stacked parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)

                # special processing for QKV projection layer
                if param_name == ".qkv_proj":
                    if mapped_name not in params_dict:
                        params_dict[mapped_name] = torch.zeros_like(loaded_weight)

                    # split loaded_weight into q,k,v three parts
                    q_weight, k_weight, v_weight = torch.chunk(loaded_weight, 3, dim=0)

                    if shard_id == "q":  # q_proj
                        params_dict[mapped_name][: q_weight.shape[0]] = q_weight
                    elif shard_id == "k":  # k_proj
                        params_dict[mapped_name][
                            q_weight.shape[0] : 2 * q_weight.shape[0]
                        ] = k_weight
                    elif shard_id == "v":  # v_proj
                        params_dict[mapped_name][2 * q_weight.shape[0] :] = v_weight
                    break

                # process LayerNorm weights
                if param_name == ".layernorm_mlp.layer_norm_weight":
                    if mapped_name in params_dict:
                        params_dict[mapped_name].data.copy_(loaded_weight)
                    break

                # combine gate_proj and up_proj of MLP
                if param_name == ".layernorm_mlp.fc1_weight":
                    if mapped_name not in params_dict:
                        params_dict[mapped_name] = torch.zeros_like(loaded_weight)

                    # split loaded_weight into gate_proj and up_proj two parts
                    gate_weight, up_weight = torch.split(
                        loaded_weight, loaded_weight.shape[0] // 2, dim=0
                    )

                    if shard_id == 0:  # gate_proj
                        params_dict[mapped_name][: gate_weight.shape[0]] = gate_weight
                    elif shard_id == 1:  # up_proj
                        params_dict[mapped_name][gate_weight.shape[0] :] = up_weight
                    break

                # process down_proj weights
                elif param_name == ".layernorm_mlp.fc2_weight":
                    if mapped_name in params_dict:
                        params_dict[mapped_name].data.copy_(loaded_weight)
                    break

                # skip extra bias of GPTQ model
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # process other normal parameters
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        # process weight binding of embedding layer
        if (
            hasattr(self.config, "tie_word_embeddings")
            and self.config.tie_word_embeddings
        ):
            param = self.lm_head.weight
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, self.model.embed_tokens.weight)

        # apply torchao configuration
        # apply_torchao_config_(self, params_dict, set(["proj.weight"])) #zhuohaol: this is the old version of apply_torchao_config_ in torchao_utils.py
        apply_torchao_config_to_model(self, params_dict, set(["proj.weight"]))


class TEPhi3ForCausalLM(TELlamaForCausalLM):
    pass


EntryClass = [TELlamaForCausalLM, TEPhi3ForCausalLM]
