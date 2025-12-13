# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class HunyuanVAEArchConfig(VAEArchConfig):
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    down_block_types: tuple[str, ...] = (
        "HunyuanVideoDownBlock3D",
        "HunyuanVideoDownBlock3D",
        "HunyuanVideoDownBlock3D",
        "HunyuanVideoDownBlock3D",
    )
    up_block_types: tuple[str, ...] = (
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
        "HunyuanVideoUpBlock3D",
    )
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    norm_num_groups: int = 32
    scaling_factor: float = 0.476986
    spatial_compression_ratio: int = 8
    temporal_compression_ratio: int = 4
    mid_block_add_attention: bool = True

    def __post_init__(self):
        self.spatial_compression_ratio: int = 2 ** (len(self.block_out_channels) - 1)


@dataclass
class HunyuanVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=HunyuanVAEArchConfig)
