# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class Hunyuan3DDiTArchConfig(DiTArchConfig):
    """Architecture config for Hunyuan3D DiT (Flux-style for Hunyuan3D-2.0)."""

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"(.*)\.img_mlp\.0\.(.*)$": r"\1.img_mlp.fc_in.\2",
            r"(.*)\.img_mlp\.2\.(.*)$": r"\1.img_mlp.fc_out.\2",
            r"(.*)\.txt_mlp\.0\.(.*)$": r"\1.txt_mlp.fc_in.\2",
            r"(.*)\.txt_mlp\.2\.(.*)$": r"\1.txt_mlp.fc_out.\2",
        }
    )

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
