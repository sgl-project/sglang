# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 video VAE (Swin3D neighborhood-attention "ViTNorm") configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class Flux3VideoVAEArchConfig(VAEArchConfig):
    z_dim: int = 96
    embed_dim: int = 256
    patch_size: tuple[int, int, int] = (1, 4, 4)
    window_size: tuple[int, int, int] = (5, 5, 5)
    enc_depths: tuple[int, ...] = (1, 4, 8, 8)
    dec_depths: tuple[int, ...] = (1, 4, 8, 8)
    num_heads: tuple[int, ...] = (4, 8, 16, 32)
    temporal: tuple[bool, ...] = (False, False, True, True)
    enc_causal: bool = True
    dec_causal: bool = False
    qk_norm: bool = True
    patch_norm: bool = False
    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 32
    # Encoding chunk length in frames; consecutive chunks overlap by one frame.
    chunk_size_frames: int = 45
    # Looped decode: every decoder block attends over temporal windows of at
    # most this many latent frames (plus the attention halo). ``None`` decodes
    # the whole latent at once.
    decoder_max_t: int | None = 8


@dataclass
class Flux3VideoVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=Flux3VideoVAEArchConfig)
