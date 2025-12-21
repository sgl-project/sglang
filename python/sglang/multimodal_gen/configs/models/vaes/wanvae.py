# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class WanVAEArchConfig(VAEArchConfig):
    base_dim: int = 96
    decoder_base_dim: int | None = None
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    latents_mean: tuple[float, ...] = (
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    )
    latents_std: tuple[float, ...] = (
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    )
    is_residual: bool = False
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int | None = None
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8
    clip_output: bool = True

    def __post_init__(self):
        self.scaling_factor: torch.tensor = 1.0 / torch.tensor(self.latents_std).view(
            1, self.z_dim, 1, 1, 1
        )
        self.shift_factor: torch.tensor = torch.tensor(self.latents_mean).view(
            1, self.z_dim, 1, 1, 1
        )
        self.temporal_compression_ratio = self.scale_factor_temporal
        self.spatial_compression_ratio = self.scale_factor_spatial


@dataclass
class WanVAEConfig(VAEConfig):
    arch_config: WanVAEArchConfig = field(default_factory=WanVAEArchConfig)
    use_feature_cache: bool = True

    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def __post_init__(self):
        self.blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        ) * 2
