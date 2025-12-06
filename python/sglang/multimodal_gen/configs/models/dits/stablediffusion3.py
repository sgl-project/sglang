# SPDX-License-Identifier: Apache-2.0
"""StableDiffusion3 Transformer model configuration"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class StableDiffusion3TransformerArchConfig(DiTArchConfig):
    """Architecture configuration for StableDiffusion3 Transformer."""

    sample_size: int = 128
    in_channels: int = 16
    out_channels: int = 16
    cross_attention_dim: int = 4096
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 2048

    _class_name: str = "SD3Transformer2DModel"


@dataclass
class StableDiffusion3TransformerConfig(DiTConfig):
    """Configuration for StableDiffusion3 Transformer model."""

    arch_config: StableDiffusion3TransformerArchConfig = field(
        default_factory=StableDiffusion3TransformerArchConfig
    )
