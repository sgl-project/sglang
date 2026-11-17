# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


@dataclass
class MingImageEncoderArchConfig(TextEncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["MingImageEncoder"])
    hidden_size: int = 2048
    num_attention_heads: int = 16
    intermediate_size: int = 5120
    llm_config: dict = field(default_factory=dict)
    vision_config: dict = field(default_factory=dict)
    connector_config: dict = field(default_factory=dict)
    projection_config: dict = field(default_factory=dict)


@dataclass
class MingImageEncoderConfig(TextEncoderConfig):
    arch_config: MingImageEncoderArchConfig = field(
        default_factory=MingImageEncoderArchConfig
    )
