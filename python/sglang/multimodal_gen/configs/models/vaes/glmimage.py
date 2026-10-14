from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class GlmImageVAEArchConfig(VAEArchConfig):
    spatial_compression_ratio: int = 1

    base_dim: int = 96
    decoder_base_dim: int | None = None
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0

    is_residual: bool = False
    input_channels: int = 3
    out_channels: int = 3
    patch_size: int | None = None
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8
    clip_output: bool = True

    scaling_factor: float | torch.Tensor = 0

    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None
    shift_factor: float | None = None
    latent_channels: int = 16
    in_channels: int = 16


@dataclass
class GlmImageVAEConfig(VAEConfig):
    arch_config: GlmImageVAEArchConfig = field(default_factory=GlmImageVAEArchConfig)

    use_feature_cache: bool = True

    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def get_vae_scale_factor(self):
        return 2 ** len(self.arch_config.temperal_downsample)

    def __post_init__(self):
        self.blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        ) * 2

    def post_init(self):
        self.arch_config.vae_scale_factor = 2 ** (
            len(self.arch_config.temperal_downsample)
        )
        self.arch_config.spatial_compression_ratio = self.arch_config.vae_scale_factor
