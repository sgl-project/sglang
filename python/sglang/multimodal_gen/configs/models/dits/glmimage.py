from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class GlmImageArchConfig(DiTArchConfig):
    patch_size: int = 2
    in_channels: int = 16
    out_channels: int | None = 16
    num_layers: int = 30
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    condition_dim: int = 256
    prior_vq_quantizer_codebook_size: int = 16384
    text_embed_dim: int = 1472
    time_embed_dim: int = 512

    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # LoRA mappings
            r"^(transformer_blocks\.\d+\.attn\..*\.lora_[AB])\.default$": r"\1",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class GlmImageDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=GlmImageArchConfig)

    prefix: str = "glmimage"
