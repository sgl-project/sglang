# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class StepVideoVAEArchConfig(VAEArchConfig):
    in_channels: int = 3
    out_channels: int = 3
    z_channels: int = 64
    num_res_blocks: int = 2
    version: int = 2
    frame_len: int = 17
    world_size: int = 1

    spatial_compression_ratio: int = 16
    temporal_compression_ratio: int = 8

    scaling_factor: float = 1.0


@dataclass
class StepVideoVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=StepVideoVAEArchConfig)
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    use_temporal_scaling_frames: bool = False
