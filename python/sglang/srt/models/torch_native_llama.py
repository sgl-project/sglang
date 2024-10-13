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
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import types
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn.parameter import Parameter
from transformers import LlamaConfig
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.torchao_utils import apply_torchao_config_
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def gate_up_proj_weight_loader(
    self,
    param: Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: Optional[int] = None,
):
    if loaded_shard_id is None:
        shard_offsets: List[Tuple[int, int, int]] = []
        for i, output_size in enumerate(self.output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size
        for shard_id, shard_offset, shard_size in shard_offsets:
            loaded_weight_shard = loaded_weight.narrow(
                output_dim, shard_offset, shard_size
            )
            self.weight_loader(param, loaded_weight_shard, shard_id)
    else:
        assert loaded_shard_id < len(self.output_sizes)
        param_data = param.data
        shard_size = loaded_weight.shape[0]
        shard_offset = loaded_shard_id * shard_size
        param_data = param_data.narrow(0, shard_offset, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
    return


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(
            hidden_size,
            intermediate_size * 2,
            bias=False,
        )
        self.gate_up_proj.output_sizes = [intermediate_size] * 2
        self.gate_up_proj.weight_loader = types.MethodType(
            gate_up_proj_weight_loader, self.gate_up_proj
        )
        self.gate_up_proj.weight.weight_loader = self.gate_up_proj.weight_loader
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


def _get_shard_offset_mapping(self, loaded_shard_id: str):
    shard_offset_mapping = {
        "q": 0,
        "k": self.num_heads * self.head_size,
        "v": (self.num_heads + self.num_kv_heads) * self.head_size,
        "total": (self.num_heads + 2 * self.num_kv_heads) * self.head_size,
    }
    return shard_offset_mapping.get(loaded_shard_id)


def _get_shard_size_mapping(self, loaded_shard_id: str):
    shard_size_mapping = {
        "q": self.num_heads * self.head_size,
        "k": self.num_kv_heads * self.head_size,
        "v": self.num_kv_heads * self.head_size,
    }
    return shard_size_mapping.get(loaded_shard_id)


def qkv_proj_weight_loader(
    self,
    param: Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: Optional[str] = None,
):
    if loaded_shard_id is None:
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
        ]
        for shard_id, shard_offset, shard_size in shard_offsets:
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader(param, loaded_weight_shard, shard_id)
    else:
        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        param_data = param.data
        param_data = param_data.narrow(0, shard_offset, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
    return


class LlamaAttention(nn.Module):
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

        self.qkv_proj = torch.nn.Linear(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.qkv_proj.total_num_heads = self.total_num_heads
        self.qkv_proj.head_size = self.head_dim
        self.qkv_proj.total_num_kv_heads = self.total_num_kv_heads
        self.qkv_proj.num_heads = self.total_num_heads
        self.qkv_proj.num_kv_heads = self.total_num_kv_heads
        self.qkv_proj.weight_loader = types.MethodType(
            qkv_proj_weight_loader, self.qkv_proj
        )
        self.qkv_proj._get_shard_offset_mapping = types.MethodType(
            _get_shard_offset_mapping, self.qkv_proj
        )
        self.qkv_proj._get_shard_size_mapping = types.MethodType(
            _get_shard_size_mapping, self.qkv_proj
        )
        self.qkv_proj.weight.weight_loader = self.qkv_proj.weight_loader
        self.qkv_proj.weight.output_dim = 0
        self.o_proj = torch.nn.Linear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
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
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
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
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = LlamaAttention(
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
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
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
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config, i, quant_config=quant_config, prefix=f"model.layers.{i}"
                )
                for i in range(config.num_hidden_layers)
            ]
        )
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


class TorchNativeLlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.torchao_config = global_server_args_dict["torchao_config"]
        self.model = LlamaModel(config, quant_config=quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head.weight, forward_batch
        )

    def get_hidden_dim(self, module_name):
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
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
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
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        if (
            hasattr(self.config, "tie_word_embeddings")
            and self.config.tie_word_embeddings
        ):
            # Tie output embedding layer to input embedding layer, to solve issues where lm_head.weight is missing
            param = self.lm_head.weight
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, self.model.embed_tokens.weight)
        apply_torchao_config_(self, params_dict, set(["proj.weight"]))


class TorchNativePhi3ForCausalLM(TorchNativeLlamaForCausalLM):
    pass


EntryClass = [TorchNativeLlamaForCausalLM, TorchNativePhi3ForCausalLM]
