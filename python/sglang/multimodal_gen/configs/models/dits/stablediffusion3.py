# SPDX-License-Identifier: Apache-2.0
"""StableDiffusion3 Transformer model configuration"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class StableDiffusion3TransformerArchConfig(DiTArchConfig):
    """Architecture configuration for StableDiffusion3 Transformer, applicable to SD3-medium, SD3.5-medium, SD3.5-large."""

    sample_size: int = 128
    patch_size: int = 2
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 18
    attention_head_dim: int = 64
    num_attention_heads: int = 18
    cross_attention_dim: int = 4096
    joint_attention_dim: int = 4096
    caption_projection_dim: int = 1152
    pooled_projection_dim: int = 2048
    pos_embed_max_size: int = 96
    dual_attention_layers: tuple[int, ...] = ()
    qk_norm: str | None = None

    _class_name: str = "SD3Transformer2DModel"


@dataclass
class StableDiffusion3TransformerConfig(DiTConfig):
    """Configuration for StableDiffusion3 Transformer model."""

    arch_config: StableDiffusion3TransformerArchConfig = field(
        default_factory=StableDiffusion3TransformerArchConfig
    )
