# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for BAGEL's SigLIP NaViT image encoder."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    ImageEncoderArchConfig,
    ImageEncoderConfig,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class BagelImageEncoderArchConfig(ImageEncoderArchConfig):
    """Architecture values stored in the official BAGEL checkpoint."""

    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
        }
    )
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 26
    num_attention_heads: int = 16
    patch_size: int = 14
    max_image_size: int = 980
    min_image_size: int = 224
    max_num_patches_per_side: int = 70
    position_embedding_rows: int = 4900
    llm_hidden_size: int = 3584
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate dimensions that are fixed by ``ema.safetensors``."""
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                "BAGEL ViT hidden_size must be divisible by num_attention_heads"
            )
        if self.max_image_size != self.patch_size * self.max_num_patches_per_side:
            raise ValueError(
                "BAGEL ViT max_image_size must equal patch_size times "
                "max_num_patches_per_side"
            )
        if self.position_embedding_rows != self.max_num_patches_per_side**2:
            raise ValueError(
                "BAGEL ViT position_embedding_rows must equal the square of "
                "max_num_patches_per_side"
            )


@dataclass
class BagelImageEncoderConfig(ImageEncoderConfig):
    """Runtime configuration for BAGEL's ViT, connector, and LLM position table."""

    arch_config: BagelImageEncoderArchConfig = field(
        default_factory=BagelImageEncoderArchConfig
    )
    prefix: str = "vit_model"
