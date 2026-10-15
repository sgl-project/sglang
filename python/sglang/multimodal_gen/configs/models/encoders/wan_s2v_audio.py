# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    AudioEncoderArchConfig,
    AudioEncoderConfig,
)


@dataclass
class WanS2VAudioEncoderArchConfig(AudioEncoderArchConfig):
    architectures: list[str] = field(default_factory=lambda: ["WanS2VAudioEncoder"])
    model_id: str = "wav2vec2-large-xlsr-53-english"
    sample_rate: int = 16000
    video_rate: int = 30
    output_hidden_states: bool = True


@dataclass
class WanS2VAudioEncoderConfig(AudioEncoderConfig):
    arch_config: WanS2VAudioEncoderArchConfig = field(
        default_factory=WanS2VAudioEncoderArchConfig
    )
