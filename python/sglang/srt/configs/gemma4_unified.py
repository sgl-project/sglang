# coding=utf-8
# Copyright 2026 The SGLang team.
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
"""Gemma 4 Unified (12B, encoder-free) model configuration.

Bundled locally so SGLang can load the model without requiring a bleeding-edge
transformers release (config.json declares transformers >= 5.10.0.dev0). Only the
top-level ``Gemma4UnifiedConfig`` is registered in ``_CONFIG_REGISTRY``; the
sub-configs are coerced from nested dicts in ``__init__``.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _default_layer_types(num_hidden_layers: int):
    # Gemma 4 Unified: every 6th layer is full attention, rest sliding.
    return [
        "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
        for i in range(num_hidden_layers)
    ]


def _default_rope_parameters():
    return {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
        },
        "sliding_attention": {
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
    }


class Gemma4UnifiedTextConfig(PretrainedConfig):
    model_type = "gemma4_unified_text"

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=3840,
        intermediate_size=15360,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=1,
        sliding_window=1024,
        layer_types=None,
        rope_parameters=None,
        max_position_embeddings=131072,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
        attention_k_eq_v=True,
        final_logit_softcapping=30.0,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=262144,
        enable_moe_block=False,
        num_experts=None,
        moe_intermediate_size=None,
        top_k_experts=None,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        use_bidirectional_attention="vision",
        initializer_range=0.02,
        use_cache=True,
        tie_word_embeddings=True,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.num_global_key_value_heads = num_global_key_value_heads
        self.sliding_window = sliding_window
        self.layer_types = layer_types or _default_layer_types(num_hidden_layers)
        self.rope_parameters = rope_parameters or _default_rope_parameters()
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_activation = hidden_activation
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_k_eq_v = attention_k_eq_v
        self.final_logit_softcapping = final_logit_softcapping
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.enable_moe_block = enable_moe_block
        self.num_experts = num_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.top_k_experts = top_k_experts
        self.num_kv_shared_layers = num_kv_shared_layers
        self.use_double_wide_mlp = use_double_wide_mlp
        self.use_bidirectional_attention = use_bidirectional_attention
        self.initializer_range = initializer_range
        self.use_cache = use_cache


class Gemma4UnifiedVisionConfig(PretrainedConfig):
    model_type = "gemma4_unified_vision"

    def __init__(
        self,
        mm_embed_dim=3840,
        mm_posemb_size=1120,
        model_patch_size=48,
        patch_size=16,
        num_soft_tokens=280,
        pooling_kernel_size=3,
        output_proj_dims=3840,
        hidden_size=3840,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mm_embed_dim = mm_embed_dim
        self.mm_posemb_size = mm_posemb_size
        self.model_patch_size = model_patch_size
        self.patch_size = patch_size
        self.num_soft_tokens = num_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.output_proj_dims = output_proj_dims
        # hidden_size mirrors mm_embed_dim; Gemma4MultimodalEmbedder reads
        # output_proj_dims first, falling back to hidden_size.
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range


class Gemma4UnifiedAudioConfig(PretrainedConfig):
    model_type = "gemma4_unified_audio"

    def __init__(
        self,
        audio_embed_dim=640,
        audio_samples_per_token=640,
        hidden_size=640,
        output_proj_dims=640,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_embed_dim = audio_embed_dim
        self.audio_samples_per_token = audio_samples_per_token
        self.hidden_size = hidden_size
        self.output_proj_dims = output_proj_dims
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range


class Gemma4UnifiedConfig(PretrainedConfig):
    model_type = "gemma4_unified"
    sub_configs = {
        "text_config": Gemma4UnifiedTextConfig,
        "vision_config": Gemma4UnifiedVisionConfig,
        "audio_config": Gemma4UnifiedAudioConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        image_token_id=258880,
        video_token_id=258884,
        audio_token_id=258881,
        boi_token_id=255999,
        eoi_token_id=258882,
        boa_token_id=256000,
        eoa_token_index=258883,
        initializer_range=0.02,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if text_config is None:
            text_config = Gemma4UnifiedTextConfig()
        elif isinstance(text_config, dict):
            text_config = Gemma4UnifiedTextConfig(**text_config)

        if vision_config is None:
            vision_config = Gemma4UnifiedVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Gemma4UnifiedVisionConfig(**vision_config)

        # audio_config may legitimately be present; unlike the 26B/31B path it is
        # never the heavy conformer here.
        if audio_config is None:
            audio_config = Gemma4UnifiedAudioConfig()
        elif isinstance(audio_config, dict):
            audio_config = Gemma4UnifiedAudioConfig(**audio_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.audio_config = audio_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.boa_token_id = boa_token_id
        # config.json spells this key "eoa_token_index"; keep both spellings so
        # downstream code reading eoa_token_id still works.
        self.eoa_token_index = eoa_token_index
        self.eoa_token_id = eoa_token_index
        self.initializer_range = initializer_range

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
