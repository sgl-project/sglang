# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from transformers import Qwen3_5TextConfig

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


@dataclass
class Qwen3_5EncoderArchConfig(TextEncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["Qwen3_5Model"])
    text_config: Qwen3_5TextConfig | dict = field(default_factory=Qwen3_5TextConfig)


@dataclass
class Qwen3_5EncoderConfig(TextEncoderConfig):
    arch_config: Qwen3_5EncoderArchConfig = field(
        default_factory=Qwen3_5EncoderArchConfig
    )

    def post_diffusers_config_update(self):
        text = self.arch_config.text_config
        if isinstance(text, dict):
            text = Qwen3_5TextConfig(**text)
            self.arch_config.text_config = text
        self.arch_config.hidden_size = text.hidden_size
        self.arch_config.num_hidden_layers = text.num_hidden_layers
        self.arch_config.num_attention_heads = text.num_attention_heads
        self.arch_config.intermediate_size = text.intermediate_size
        self.arch_config.vocab_size = text.vocab_size
