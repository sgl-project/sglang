# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import List

from sglang.multimodal_gen.configs.models.vocoder.base import (
    VocoderArchConfig,
    VocoderConfig,
)


@dataclass
class LTXVocoderArchConfig(VocoderArchConfig):
    # Architecture params
    in_channels: int = 128
    hidden_channels: int = 1024
    out_channels: int = 2
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    upsample_factors: List[int] = field(default_factory=lambda: [6, 5, 2, 2, 2])
    resnet_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resnet_dilations: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    leaky_relu_negative_slope: float = 0.1
    sample_rate: int = 24000


@dataclass
class LTXVocoderConfig(VocoderConfig):
    arch_config: LTXVocoderArchConfig = field(default_factory=LTXVocoderArchConfig)
