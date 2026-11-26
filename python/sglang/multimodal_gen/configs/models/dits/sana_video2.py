# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class SanaVideo2ArchConfig(DiTArchConfig):
    input_size: int = 15
    patch_size: tuple[int, int, int] = (1, 1, 1)
    in_channels: int = 128
    hidden_size: int = 2560
    depth: int = 32
    num_heads: int = 20
    linear_head_dim: int = 128
    softmax_head_dim: int = 256
    softmax_ratio: float = 0.25
    mlp_ratio: float = 4.0
    caption_channels: int = 2304
    model_max_length: int = 300
    qk_norm: bool = True
    cross_norm: bool = True
    y_norm: bool = True
    y_norm_scale_factor: float = 0.01
    norm_eps: float = 1e-5
    attn_res_block_size: int = 8
    timestep_norm_scale_factor: float = 1.0
    fp32_attention: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.patch_size = tuple(self.patch_size)
        if self.patch_size != (1, 1, 1):
            raise ValueError("SANA-Video 2.0 requires patch_size=(1, 1, 1)")
        if self.depth < 1 or self.attn_res_block_size < 1:
            raise ValueError("depth and attn_res_block_size must be positive")
        if not 0 < self.softmax_ratio <= 1:
            raise ValueError("softmax_ratio must be in (0, 1]")
        if any(
            self.hidden_size % dim
            for dim in (self.linear_head_dim, self.softmax_head_dim, self.num_heads)
        ):
            raise ValueError("hidden_size must divide evenly into attention heads")
        self.num_attention_heads = self.num_heads
        self.num_channels_latents = self.in_channels


@dataclass
class SanaVideo2Config(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=SanaVideo2ArchConfig)
    prefix: str = "SanaVideo2"
