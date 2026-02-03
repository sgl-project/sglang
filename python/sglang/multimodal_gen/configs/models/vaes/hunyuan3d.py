# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class Hunyuan3DVAEArchConfig(VAEArchConfig):
    """Architecture config for Hunyuan3D VAE."""

    latent_shape: tuple[int, ...] = (1024, 64)
    scale_factor: float = 1.0


@dataclass
class Hunyuan3DVAEConfig(VAEConfig):
    """VAE configuration for Hunyuan3D."""

    arch_config: Hunyuan3DVAEArchConfig = field(default_factory=Hunyuan3DVAEArchConfig)
    subfolder: str = "hunyuan3d-dit-v2-0"
    load_encoder: bool = False
    load_decoder: bool = True
