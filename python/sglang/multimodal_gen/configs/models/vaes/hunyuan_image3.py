from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class HunyuanImage3VAEArchConfig(VAEArchConfig):
    """VAE architecture配置 for HunyuanImage-3."""

    spatial_compression_ratio: int = 16

    # VAE architecture params (from HF config)
    block_out_channels: tuple[int, ...] = (128, 256, 512, 1024, 1024)
    layers_per_block: int = 2
    latent_channels: int = 32
    in_channels: int = 3
    out_channels: int = 3

    # Scaling factors
    scaling_factor: float = 0.562679178327931
    ffactor_spatial: int = 16
    ffactor_temporal: int = 4

    # Latent statistics (optional, for normalization)
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None
    shift_factor: float | None = None

    # Temporal downsampling
    temperal_downsample: tuple[bool, ...] = (False, True, True)

    # Other params
    is_residual: bool = False
    patch_size: int | None = None
    clip_output: bool = True


@dataclass
class HunyuanImage3VAEConfig(VAEConfig):
    """VAE config for HunyuanImage-3."""

    arch_config: HunyuanImage3VAEArchConfig = field(
        default_factory=HunyuanImage3VAEArchConfig
    )

    use_feature_cache: bool = False
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def get_vae_scale_factor(self):
        return self.arch_config.ffactor_spatial

    def __post_init__(self):
        self.blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        ) * 2

    def post_init(self):
        self.arch_config.vae_scale_factor = self.arch_config.ffactor_spatial
        self.arch_config.spatial_compression_ratio = self.arch_config.ffactor_spatial
