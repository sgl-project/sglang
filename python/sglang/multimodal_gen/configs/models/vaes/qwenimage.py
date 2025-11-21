# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig
from sglang.multimodal_gen.utils import calculate_dimensions


@dataclass
class QwenImageVAEArchConfig(VAEArchConfig):
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
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int | None = None
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8
    clip_output: bool = True

    def __post_init__(self):
        self.vae_scale_factor = 2 ** len(self.temperal_downsample)


@dataclass
class QwenImageVAEConfig(VAEConfig):
    arch_config: QwenImageVAEArchConfig = field(default_factory=QwenImageVAEArchConfig)

    use_feature_cache: bool = True

    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def calculate_dimensions(self, image, vae_scale_factor, width, height):
        width = image.size[0]
        height = image.size[1]
        width, height, _ = calculate_dimensions(1024 * 1024, width / height)
        return width, height

    def __post_init__(self):
        self.blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        ) * 2

    def post_init(self):
        self.arch_config.vae_scale_factor = 2 ** (
            len(self.arch_config.temperal_downsample)
        )
        self.arch_config.spatial_compression_ratio = self.arch_config.vae_scale_factor
