# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class OvisImageArchConfig(DiTArchConfig):
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = None
    num_layers: int = 6
    num_single_layers: int = 27
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 2048
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)

    def __post_init__(self):
        super().__post_init__()
        if self.in_channels % 4:
            raise ValueError("Ovis-Image packed latent channels must be divisible by 4")
        if any(dim <= 0 or dim % 2 for dim in self.axes_dims_rope):
            raise ValueError(
                "Ovis-Image rotary axis dimensions must be positive and even"
            )
        if sum(self.axes_dims_rope) != self.attention_head_dim:
            raise ValueError(
                "Ovis-Image rotary dimensions must sum to the head dimension"
            )
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.in_channels // 4


@dataclass
class OvisImageConfig(DiTConfig):
    arch_config: OvisImageArchConfig = field(default_factory=OvisImageArchConfig)
    prefix: str = "ovis_image"
