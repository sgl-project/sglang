# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Optional

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class Hunyuan3DDiTArchConfig(DiTArchConfig):
    """Architecture config for Hunyuan3D DiT (Flux-style for Hunyuan3D-2.0)."""

    in_channels: int = 64
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_layers: int = 16
    num_single_layers: int = 32
    mlp_ratio: float = 4.0
    context_in_dim: int = 1536
    axes_dim: tuple[int, ...] = (64,)
    theta: int = 10000
    qkv_bias: bool = True
    guidance_embed: bool = False
    time_factor: float = 1000.0

    def __post_init__(self) -> None:
        if self.num_channels_latents == 0:
            self.num_channels_latents = self.in_channels
        super().__post_init__()


@dataclass
class Hunyuan3DDiTConfig(DiTConfig):
    """DiT configuration for Hunyuan3D shape generation (Flux-style)."""

    arch_config: Hunyuan3DDiTArchConfig = field(default_factory=Hunyuan3DDiTArchConfig)
    subfolder: str = "hunyuan3d-dit-v2-0"


@dataclass
class Hunyuan3DPlainDiTArchConfig(DiTArchConfig):
    """Architecture config for Hunyuan3D Plain DiT (with MoE)."""

    input_size: int = 1024
    in_channels: int = 4
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_layers: int = 24
    context_dim: int = 1024
    mlp_ratio: float = 4.0
    text_len: int = 257
    num_moe_layers: int = 6
    num_experts: int = 8
    moe_top_k: int = 2
    norm_type: str = "layer"
    qk_norm_type: str = "rms"
    qk_norm: bool = False
    qkv_bias: bool = True
    with_decoupled_ca: bool = False
    additional_cond_hidden_state: int = 768
    decoupled_ca_dim: int = 16
    decoupled_ca_weight: float = 1.0
    use_pos_emb: bool = False
    use_attention_pooling: bool = True
    guidance_cond_proj_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_channels_latents == 0:
            self.num_channels_latents = self.in_channels
        super().__post_init__()


@dataclass
class Hunyuan3DPlainDiTConfig(DiTConfig):
    """DiT configuration for Hunyuan3D Plain model (with MoE)."""

    arch_config: Hunyuan3DPlainDiTArchConfig = field(
        default_factory=Hunyuan3DPlainDiTArchConfig
    )
