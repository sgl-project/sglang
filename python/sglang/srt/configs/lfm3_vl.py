# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""LFM3-VL configuration with SGLang's hybrid language-model cache properties.

HF converts legacy flat checkpoints into nested text/vision configs. Reuse that
conversion and delegate cache metadata to the extended LFM3 text config.
"""

from sglang.srt.configs.lfm3 import Lfm3Config, Lfm3ServingConfig
from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import Mamba2CacheParams


class Lfm3VlConfig(Lfm3ServingConfig):
    model_type = "lfm3_vl"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if not isinstance(self.text_config, Lfm3Config):
            self.text_config = Lfm3Config.from_hf_config(self.text_config)

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return self.text_config.full_attention_layer_ids

    @property
    def linear_layer_ids(self) -> list[int]:
        return self.text_config.linear_layer_ids

    @property
    def mamba_chunk_size(self) -> int:
        return self.text_config.mamba_chunk_size

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams | None:
        return self.text_config.mamba2_cache_params


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=Lfm3VlConfig,
        arch_names=["Lfm3VlForConditionalGeneration"],
        backend_class_name=(
            "sglang.srt.layers.attention.hybrid_linear_attn_backend.Mamba2AttnBackend"
        ),
        support_mamba_cache_extra_buffer=True,
    )
)
