# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class ZImageArchConfig(DiTArchConfig):
    all_patch_size: Tuple[int, ...] = (2,)
    all_f_patch_size: Tuple[int, ...] = (1,)
    in_channels: int = 16
    out_channels: int | None = None
    dim: int = 3840
    num_layers: int = 30
    n_refiner_layers: int = 2
    num_attention_heads: int = 30
    n_kv_heads: int = 30
    norm_eps: float = 1e-5
    qk_norm: bool = True
    cap_feat_dim: int = 2560
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: Tuple[int, int, int] = (32, 48, 48)
    axes_lens: Tuple[int, int, int] = (1024, 512, 512)

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.num_channels_latents = self.in_channels
        self.hidden_size = self.dim


@dataclass
class ZImageDitConfig(DiTConfig):
    arch_config: ZImageArchConfig = field(default_factory=ZImageArchConfig)

    prefix: str = "zimage"
