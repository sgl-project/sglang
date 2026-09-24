# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.stablediffusion3 import (
    StableDiffusion3VAEArchConfig,
    StableDiffusion3VAEConfig,
)


@dataclass
class BooguImageVAEArchConfig(StableDiffusion3VAEArchConfig):
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    sample_size: int = 1024


@dataclass
class BooguImageVAEConfig(StableDiffusion3VAEConfig):
    arch_config: BooguImageVAEArchConfig = field(
        default_factory=BooguImageVAEArchConfig
    )
