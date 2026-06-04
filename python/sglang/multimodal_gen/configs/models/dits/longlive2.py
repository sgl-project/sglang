# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/NVlabs/LongLive

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)

@dataclass
class LongLive2ArchConfig(WanVideoArchConfig):
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    ffn_dim: int = 14336
    num_layers: int = 30
    local_attn_size: int = 32
    sink_size: int = 8
    num_frames_per_block: int = 8
    sliding_window_num_frames: int = 32

@dataclass
class LongLive2VideoConfig(WanVideoConfig):
    arch_config: DiTArchConfig = field(default_factory=LongLive2ArchConfig)
