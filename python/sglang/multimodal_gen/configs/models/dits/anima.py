# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class AnimaArchConfig(DiTArchConfig):
    in_channels: int = 16
    out_channels: int = 16
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    num_layers: int = 28
    mlp_ratio: float = 4.0
    text_embed_dim: int = 1024
    adaln_lora_dim: int = 256
    patch_size: tuple[int, int, int] = (1, 2, 2)
    max_size: tuple[int, int, int] = (128, 240, 240)
    rope_scale: tuple[float, float, float] = (1.0, 4.0, 4.0)
    concat_padding_mask: bool = True
    extra_pos_embed_type: str | None = None
    use_crossattn_projection: bool = False
    img_context_dim_in: int | None = None

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.in_channels


@dataclass
class AnimaDiTConfig(DiTConfig):
    arch_config: AnimaArchConfig = field(default_factory=AnimaArchConfig)
    prefix: str = "anima"
