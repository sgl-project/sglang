# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class SanaArchConfig(DiTArchConfig):
    patch_size: int = 1
    in_channels: int = 32
    out_channels: int = 32
    num_layers: int = 20
    attention_head_dim: int = 32
    num_attention_heads: int = 70
    num_cross_attention_heads: int = 20
    cross_attention_head_dim: int = 112
    cross_attention_dim: int = 2240
    caption_channels: int = 2304

    mlp_ratio: float = 2.5
    # "rms_norm_across_heads" applies RMSNorm over the full (num_heads * head_dim)

    qk_norm: str = "rms_norm_across_heads"
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-6
    sample_size: int = 32
    guidance_embeds: bool = False

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^transformer\.(.*)$": r"\1",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class SanaConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=SanaArchConfig)
    prefix: str = "Sana"
