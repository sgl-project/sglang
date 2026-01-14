from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class LTX2ArchConfig(DiTArchConfig):
    in_channels: int = 128
    out_channels: int = 128
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    num_layers: int = 28
    cross_attention_dim: int = 4096
    # In LTX-2, text encoder features are projected to modality dims.
    # ltx-core defaults to 3840 here.
    caption_channels: int = 3840

    # Audio specific
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_cross_attention_dim: int = 2048

    # Compression ratios needed for RoPE coordinate calculation
    vae_temporal_compression_ratio: int = 8
    vae_spatial_compression_ratio: int = 32

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"model\.diffusion_model\.(.*)$": r"\1",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class LTX2Config(DiTConfig):

    arch_config: DiTArchConfig = field(default_factory=LTX2ArchConfig)

    prefix: str = "LTX2"
