# SPDX-License-Identifier: Apache-2.0
"""Config for the FLUX.3 image VAE (the FLUX.2 packed image autoencoder,
``AutoEncoder``). Ported from the BFL FLUX.3 reference
(``f3.models.image_vae.autoencoder``). Encode packs a 2x2 spatial block into the
channel dim, so latents are 4*z_channels (=128) at H/16, W/16.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class Flux3ImageVAEArchConfig(VAEArchConfig):
    temporal_compression_ratio: int = 1
    spatial_compression_ratio: int = 16  # 8x conv downsample x 2x2 pixel-shuffle
    scaling_factor: float = 1.0

    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32
    strided_attention: bool = True


@dataclass
class Flux3ImageVAEConfig(VAEConfig):
    arch_config: Flux3ImageVAEArchConfig = field(
        default_factory=Flux3ImageVAEArchConfig
    )
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    use_parallel_decode: bool = False

    def get_vae_scale_factor(self) -> int:
        return self.arch_config.spatial_compression_ratio
