# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Configuration for the exported LFM3 Omni input-to-text model."""

from sglang.srt.configs.lfm3 import Lfm3Config, Lfm3ServingConfig
from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)


class Lfm3OmniConfig(Lfm3ServingConfig):
    model_type = "lfm3_omni"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if not isinstance(self.text_config, Lfm3Config):
            self.text_config = Lfm3Config.from_hf_config(self.text_config)
        eos = self.text_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        # Native text-only generation ends at either the turn or text marker.
        self.eos_token_id = list(dict.fromkeys([*eos_ids, self.text_end_token_id]))
        self.pad_token_id = self.text_config.pad_token_id

    @property
    def audio_config(self):
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
        config_class=Lfm3OmniConfig,
        arch_names=["Lfm3OmniForConditionalGeneration"],
        backend_class_name="sglang.srt.layers.attention.hybrid_linear_attn_backend.Mamba2AttnBackend",
        support_mamba_cache_extra_buffer=True,
    )
)
