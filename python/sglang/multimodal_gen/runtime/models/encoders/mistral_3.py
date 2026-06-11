# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
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

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.mistral3 import (
    Mistral3EncoderConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
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
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class MistralMLP(nn.Module):
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
                "Only silu is supported for Mistral."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class MistralAttention(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tp_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # Mistral exposes an explicit head_dim (introduced by Mistral-Nemo).
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
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

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            rope_scaling=rope_scaling,
            is_neox_style=True,
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

        batch_size = q.shape[0]
        seq_len = q.shape[1]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        output, _ = self.o_proj(attn_output)
        return output


class MistralDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get("rope_theta", 10000.0)
        rope_scaling = rope_parameters or None
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False)
        bias_o_proj = attention_bias

        self.self_attn = MistralAttention(
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
        self.mlp = MistralMLP(
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
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MistralModel(nn.Module):
    """TP-parallel Mistral decoder stack used as a text encoder."""

    def __init__(self, config: Mistral3EncoderConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=getattr(config, "quant_config", None),
        )

        self.layers = nn.ModuleList(
            [
                MistralDecoderLayer(
                    config=config,
                    quant_config=getattr(config, "quant_config", None),
                    prefix=f"{prefix}.layers.{i}",
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
        output_hidden_states: bool | None = True,
        **kwargs,
    ) -> BaseEncoderOutput:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                0, hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)

        residual: torch.Tensor | None = None
        collect_hidden = bool(output_hidden_states)
        all_hidden_states: tuple[torch.Tensor, ...] | None = (
            () if collect_hidden else None
        )
        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (
                    (hidden_states,)
                    if residual is None
                    else (hidden_states + residual,)
                )
            hidden_states, residual = layer(position_ids, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class Mistral3Model(nn.Module):
    """Module-layout wrapper: exposes `language_model.*` under `model.*`."""

    def __init__(self, config: Mistral3EncoderConfig, prefix: str = "") -> None:
        super().__init__()
        self.language_model = MistralModel(config, prefix=f"{prefix}.language_model")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def forward(self, *args, **kwargs) -> BaseEncoderOutput:
        return self.language_model(*args, **kwargs)


# Scalars that may live under Mistral3Config.text_config and must be hoisted.
_HOISTED_TEXT_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "tie_word_embeddings",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "attention_bias",
    "mlp_bias",
    "sliding_window",
)


def _hoist_text_config(arch_config) -> None:
    """Lift nested HF Mistral3Config.text_config scalars onto arch_config."""
    text_config = getattr(arch_config, "text_config", None)
    if text_config is None:
        return
    for field_name in _HOISTED_TEXT_CONFIG_FIELDS:
        if hasattr(text_config, field_name):
            value = getattr(text_config, field_name)
            if value is not None:
                setattr(arch_config, field_name, value)
    rope_theta = getattr(text_config, "rope_theta", None)
    if rope_theta is not None:
        rope_params = dict(getattr(arch_config, "rope_parameters", None) or {})
        rope_params["rope_theta"] = float(rope_theta)
        rope_scaling = getattr(text_config, "rope_scaling", None)
        if rope_scaling:
            rope_params.update(rope_scaling)
        arch_config.rope_parameters = rope_params


class Mistral3ForConditionalGeneration(TextEncoder, LayerwiseOffloadableModuleMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    uses_sglang_forward_context = True
    layerwise_offload_dit_group_enabled = False
    layer_names = ["model.language_model.layers"]
    _supported_attention_backends = (
        Mistral3EncoderConfig()._supported_attention_backends
    )

    def __init__(self, config: Mistral3EncoderConfig) -> None:
        super().__init__(config)
        _hoist_text_config(config.arch_config)
        self.model = Mistral3Model(config, prefix="model")

    @property
    def language_model(self) -> MistralModel:
        return self.model.language_model

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = True,
        **kwargs,
    ) -> BaseEncoderOutput:
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=(
                True if output_hidden_states is None else output_hidden_states
            ),
            **kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        stacked_params_mapping = self.config.arch_config.stacked_params_mapping
        for name, loaded_weight in weights:
            name_lower = name.lower()
            if (
                "vision" in name_lower
                or "multi" in name_lower
                or "lm_head" in name_lower
            ):
                continue
            name = name.replace("language_model.model.", "model.language_model.")
            if "rotary_emb.inv_freq" in name:
                continue
            if "scale" in name:
                kv_scale_name = maybe_remap_kv_scale_name(name, params_dict)
                if kv_scale_name is None:
                    continue
                name = kv_scale_name

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                merged_name = name.replace(weight_name, param_name)
                if merged_name.endswith(".bias") and merged_name not in params_dict:
                    break
                if merged_name not in params_dict:
                    break
                param = params_dict[merged_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(merged_name)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning("Param %s from weight is not loaded", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params


EntryClass = Mistral3ForConditionalGeneration
