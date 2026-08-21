# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class ErnieImageVAEArchConfig(VAEArchConfig):
    spatial_compression_ratio: int = 8

    base_dim: int = 96
    decoder_base_dim: int | None = None
    z_dim: int = 32
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0

    is_residual: bool = False
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int | None = None
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8
    clip_output: bool = True


@dataclass
class ErnieImageVAEConfig(VAEConfig):
    arch_config: ErnieImageVAEArchConfig = field(
        default_factory=ErnieImageVAEArchConfig
    )

    use_feature_cache: bool = True

    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def get_vae_scale_factor(self):
        # 8 spatial compression (VAE) * 2 patch = 16 total, consistent with pipeline config
        return self.arch_config.scale_factor_spatial

    def __post_init__(self):
        self.blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        ) * 2

    def post_init(self):
        if self.arch_config.dim_mult:
            self.arch_config.vae_scale_factor = 2 ** (
                len(self.arch_config.dim_mult) - 1
            )
        else:
            self.arch_config.vae_scale_factor = self.arch_config.scale_factor_spatial
        self.arch_config.spatial_compression_ratio = self.arch_config.vae_scale_factor
