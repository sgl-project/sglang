# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.base import ArchConfig, ModelConfig


@dataclass
class VocoderArchConfig(ArchConfig):
    in_channels: int = 128
    hidden_channels: int = 1024
    out_channels: int = 2


@dataclass
class VocoderConfig(ModelConfig):
    arch_config: VocoderArchConfig = field(default_factory=VocoderArchConfig)
