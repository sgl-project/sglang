from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class LongCatImageArchConfig(DiTArchConfig):
    patch_size: int = 1
    in_channels: int = 64  # packed: 16 * 4
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 3584
    pooled_projection_dim: int = 3584
    axes_dims_rope: list = field(default_factory=lambda: [16, 56, 56])

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = 16  # unpacked channels


@dataclass
class LongCatImageDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LongCatImageArchConfig)

    prefix: str = "longcat_image"
