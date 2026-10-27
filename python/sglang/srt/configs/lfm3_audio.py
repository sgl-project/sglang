# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Configuration for the LFM3 audio-input/text-output serving model."""

from sglang.srt.configs.lfm3 import Lfm3Config, Lfm3ServingConfig
from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from sglang.srt.multimodal.lfm3_audio import Lfm3AudioProcessor


@register_customized_processor(Lfm3AudioProcessor)
class Lfm3AudioConfig(Lfm3ServingConfig):
    model_type = "lfm3_audio"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if not isinstance(self.text_config, Lfm3Config):
            self.text_config = Lfm3Config.from_hf_config(self.text_config)
        if getattr(self, "eos_token_id", None) is None:
            self.eos_token_id = self.text_config.eos_token_id
        if getattr(self, "pad_token_id", None) is None:
            self.pad_token_id = self.text_config.pad_token_id

    @property
    def audio_config(self):
        # Signals audio understanding to the OpenAI serving path.
        return self.audio_encoder_config

    @property
    def full_attention_layer_ids(self):
        return self.text_config.full_attention_layer_ids

    @property
    def linear_layer_ids(self):
        return self.text_config.linear_layer_ids

    @property
    def mamba_chunk_size(self):
        return self.text_config.mamba_chunk_size

    @property
    def mamba2_cache_params(self):
        return self.text_config.mamba2_cache_params


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=Lfm3AudioConfig,
        arch_names=["Lfm3AudioForConditionalGeneration"],
        backend_class_name=(
            "sglang.srt.layers.attention.hybrid_linear_attn_backend.Mamba2AttnBackend"
        ),
        support_mamba_cache_extra_buffer=True,
    )
)
