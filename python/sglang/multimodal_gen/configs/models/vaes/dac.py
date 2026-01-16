# Copied and adapted from: mossVG/mova/diffusion/models/dac_vae.py
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class DacVAEArchConfig(VAEArchConfig):
    sample_rate: int = 44100
    hop_length: int = 2048
    latent_dim: int = 128


@dataclass
class DacVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=DacVAEArchConfig)
    load_encoder: bool = False
    load_decoder: bool = True
