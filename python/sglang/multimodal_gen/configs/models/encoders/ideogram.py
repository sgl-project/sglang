# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.qwen3vl import (
    Qwen3VLArchConfig,
    Qwen3VLConfig,
)


@dataclass
class Ideogram4TextEncoderConfig(Qwen3VLConfig):
    """Use the local Ideogram text_encoder as a language-only Qwen3-VL encoder."""

    def update_model_arch(self, source_model_dict):
        super().update_model_arch(source_model_dict)
        self.post_diffusers_config_update()

    def post_diffusers_config_update(self):
        self.arch_config.architectures = ["IdeogramQwen3VLTextEncoder"]
        self.arch_config.ideogram_fp8_weight_only = True

    def finalize_model_arch(self):
        self.post_diffusers_config_update()

    arch_config: Qwen3VLArchConfig = field(default_factory=Qwen3VLArchConfig)
