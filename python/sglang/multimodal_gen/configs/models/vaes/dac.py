# Copied and adapted from: mossVG/mova/diffusion/models/dac_vae.py
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List

from sglang.multimodal_gen.configs.models.base import ArchConfig, ModelConfig


@dataclass
class DacVAEArchConfig(ArchConfig):
    codebook_dim: int = 8
    codebook_size: int = 1024
    continuous: bool = True
    decoder_dim: int = 2048
    decoder_rates: List[int] = field(default_factory=lambda: [8, 5, 4, 3, 2])
    encoder_dim: int = 128
    encoder_rates: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 8])
    hop_length: int = 3840
    latent_dim: int = 128
    n_codebooks: int = 9
    quantizer_dropout: bool = False
    sample_rate: int = 48000


@dataclass
class DacVAEConfig(ModelConfig):
    arch_config: DacVAEArchConfig = field(default_factory=DacVAEArchConfig)
    load_encoder: bool = True
    load_decoder: bool = True
