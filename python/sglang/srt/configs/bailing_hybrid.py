# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""BailingHybrid model configuration"""

import copy
import enum
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateShape,
)
from sglang.srt.configs.qwen3_vl import Qwen3VLMoeVisionConfig
from sglang.srt.runtime_context import get_parallel

logger = logging.get_logger(__name__)


def is_bailing_multi_gate_enabled(config: PretrainedConfig) -> bool:
    """Select MultiRouter only when the checkpoint config declares it."""
    return (
        bool(getattr(config, "multi_gate", False))
        or getattr(config, "router_type", "topN") == "MultiRouter"
    )


class HybridLayerType(enum.Enum):
    full_attention = "attention"
    linear_attention = "linear_attention"


class BailingHybridConfig(PretrainedConfig):
    model_type = "bailing_hybrid"
    ignore_keys_at_rope_validation = {"mrope_section", "video_rope"}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=157184,
        hidden_size=2048,
        intermediate_size=5120,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        use_qkv_bias=False,  # bailing only
        use_bias=False,  # bailing only
        rms_norm_eps=1e-06,
        tie_word_embeddings=False,  # PretrainedConfig key, here change default value.
        embedding_dropout=0.0,
        attention_dropout=0.0,
        output_dropout=0.0,
        initializer_range=0.02,
        max_position_embeddings=32768,
        rope_theta=600000.0,
        use_cache=True,
        max_window_layers=20,
        rope_scaling=None,
        pad_token_id=156892,
        eos_token_id=156892,
        num_experts=256,
        num_shared_experts=1,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        moe_intermediate_size=512,
        first_k_dense_replace=1,
        head_dim=128,
        output_router_logits=False,
        use_qk_norm=True,
        num_nextn_predict_layers=0,
        mtp_loss_scaling_factor=0,
        moe_router_enable_expert_bias=False,
        routed_scaling_factor=1.0,
        layer_group_size=1,
        group_norm_size=1,
        linear_silu=False,
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        rope_interleave=True,
        # KDA (Ling-V3) linear-attention variant. Absent from a V2.5 /
        # lightning checkpoint, which keeps the Mamba2 branch below.
        short_conv_kernel_size=None,
        no_kda_lora=False,
        kda_safe_gate=False,
        kda_lower_bound=None,
        # NoPE MLA: the rope half of the query/key is dropped entirely.
        use_mla_nope=False,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.routed_scaling_factor = routed_scaling_factor

        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits

        self.layer_group_size = layer_group_size
        self.group_norm_size = group_norm_size
        self.linear_silu = linear_silu
        self.num_linear_key_value_heads = num_attention_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.rope_interleave = rope_interleave
        self.short_conv_kernel_size = short_conv_kernel_size
        # KDA is what distinguishes Ling-V3 from the V2.5 / lightning
        # checkpoints; only the former carries a short conv.
        self.use_kda = short_conv_kernel_size is not None
        self.no_kda_lora = no_kda_lora
        self.kda_safe_gate = kda_safe_gate
        self.kda_lower_bound = kda_lower_bound if kda_safe_gate else None
        self.use_mla_nope = use_mla_nope
        self.for_nextn_model = False
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        if self.for_nextn_model:
            return [HybridLayerType.full_attention.value]

        layer_type_list = []

        if isinstance(self.layer_group_size, int):
            for l in range(self.num_hidden_layers):
                if (l + 1) % self.layer_group_size == 0:
                    layer_type_list.append(HybridLayerType.full_attention.value)
                else:
                    layer_type_list.append(HybridLayerType.linear_attention.value)
        else:
            # Per-layer schedule: 1 marks a linear-attention layer.
            assert len(self.layer_group_size) == self.num_hidden_layers, (
                "When layer_group_size is a list, its length must be equal to num_hidden_layers"
            )
            for l in range(self.num_hidden_layers):
                if self.layer_group_size[l] == 1:
                    layer_type_list.append(HybridLayerType.linear_attention.value)
                else:
                    layer_type_list.append(HybridLayerType.full_attention.value)

        return layer_type_list

    @property
    def linear_layer_ids(self):
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.linear_attention.value
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            i
            for i, type_value in enumerate(self.layers_block_type)
            if type_value == HybridLayerType.full_attention.value
        ]

    @property
    def mamba2_cache_params(self) -> Union[KimiLinearCacheParams, Mamba2CacheParams]:

        if self.use_kda:
            shape = KimiLinearStateShape.create(
                tp_world_size=get_parallel().attn_tp_size,
                num_heads=self.num_attention_heads,
                head_dim=self.head_dim,
                conv_kernel_size=self.short_conv_kernel_size,
            )

            return KimiLinearCacheParams(shape=shape, layers=self.linear_layer_ids)

        shape = Mamba2StateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            intermediate_size=0,
            n_groups=0,
            num_heads=self.num_linear_key_value_heads,
            head_dim=self.head_dim,
            state_size=self.head_dim,
            conv_kernel=1,
        )

        return Mamba2CacheParams(shape=shape, layers=self.linear_layer_ids)


def _resolve_vl_token_id(value, text_config, legacy_name: str, default: int) -> int:
    if value is not None:
        return value
    legacy_value = getattr(text_config, legacy_name, None)
    return default if legacy_value is None else legacy_value


class BailingMoeV3VLConfig(PretrainedConfig):
    model_type = "bailing_moe_v3_vl"
    ignore_keys_at_rope_validation = {"mrope_section", "video_rope"}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=None,
        video_token_id=None,
        vision_start_token_id=None,
        vision_end_token_id=157159,
        tie_word_embeddings=False,
        mrope_section=None,
        video_start_token_id=None,
        video_end_token_id=157161,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            vision_config = dict(vision_config)
            # The public Bailing checkpoint omits deepstack entirely. Do not
            # inherit Qwen3-VL's architecture-specific deepstack defaults.
            vision_config.setdefault("deepstack_visual_indexes", [])
            vision_config = Qwen3VLMoeVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen3VLMoeVisionConfig(deepstack_visual_indexes=[])

        if isinstance(text_config, dict):
            text_config = BailingHybridConfig(**text_config)
        elif text_config is None:
            text_config = BailingHybridConfig()

        self.vision_config = vision_config
        self.text_config = text_config
        # Resolve once so the processor and mRoPE read the same token ids.
        self.image_token_id = _resolve_vl_token_id(
            image_token_id, text_config, "image_patch_token", 157157
        )
        self.video_token_id = _resolve_vl_token_id(
            video_token_id, text_config, "video_patch_token", 156909
        )
        self.vision_start_token_id = _resolve_vl_token_id(
            vision_start_token_id, text_config, "image_start_token", 157158
        )
        self.video_start_token_id = _resolve_vl_token_id(
            video_start_token_id, text_config, "video_start_token", 157160
        )
        self.vision_end_token_id = vision_end_token_id
        self.video_end_token_id = video_end_token_id

        if mrope_section is None:
            mrope_section = getattr(text_config, "mrope_section", None)
        if mrope_section is None:
            mrope_section = text_config.rope_parameters.get(
                "mrope_section", [8, 12, 12]
            )
        self.mrope_section = mrope_section
        text_config.rope_parameters.setdefault("rope_type", "default")
        text_config.rope_parameters.update(
            mrope_section=mrope_section,
            video_rope=True,
        )

        if self.text_config.architectures is None:
            self.text_config.architectures = ["BailingMoeV3ForCausalLM"]

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_nextn_config(self):
        config = copy.deepcopy(self.text_config)
        config.architectures = ["BailingMoeForCausalLMNextN"]
        config.for_nextn_model = True
        config.is_vl_nextn = True
        config.quantization_config = copy.deepcopy(self.quantization_config)
        config.quantization_config.setdefault("modules_to_not_convert", []).append(
            f"model.layers.{config.num_hidden_layers}.eh_proj"
        )
        config.expert_swiglu_limit_list = config.expert_swiglu_limit_list[
            config.num_hidden_layers :
        ]
        config.share_expert_swiglu_limit_list = config.share_expert_swiglu_limit_list[
            config.num_hidden_layers :
        ]
        return config

    @property
    def max_position_embeddings(self):
        return self.text_config.max_position_embeddings

    @max_position_embeddings.setter
    def max_position_embeddings(self, value):
        self.text_config.max_position_embeddings = value

    @property
    def rope_parameters(self):
        return self.text_config.rope_parameters

    @rope_parameters.setter
    def rope_parameters(self, value):
        if value is not None:
            self.text_config.rope_parameters.update(value)

    @property
    def rope_scaling(self):
        """The language model's effective RoPE parameters (v5 backcompat alias)."""
        return self.text_config.rope_parameters

    @rope_scaling.setter
    def rope_scaling(self, value):
        # A top-level rope_scaling override (e.g. --json-model-override-args
        # '{"rope_scaling": ...}') targets the language model's rope. Merge it
        # into the text config's rope_parameters so the mrope_section and
        # video_rope markers injected above survive the override.
        if isinstance(value, dict) and hasattr(self, "text_config"):
            self.text_config.rope_parameters.update(value)
        else:
            PretrainedConfig.rope_scaling.fset(self, value)
