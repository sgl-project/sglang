# Copied and adapted from: https://github.com/meituan-longcat/LongCat-AudioDiT
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from transformers import PreTrainedConfig
from transformers.models.umt5.configuration_umt5 import UMT5Config

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


class LongCatAudioDiTVaeConfig(PreTrainedConfig):
    """WAV-VAE: encoder 128-d, bottleneck 64-d, hop 2048 samples/frame."""

    model_type = "audiodit_vae"

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 128,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        latent_dim: int = 64,
        encoder_latent_dim: int = 128,
        use_snake: bool = True,
        downsample_shortcut: str = "averaging",
        upsample_shortcut: str = "duplicating",
        out_shortcut: str = "averaging",
        in_shortcut: str = "duplicating",
        final_tanh: bool = False,
        downsampling_ratio: int = 2048,
        sample_rate: int = 24000,
        scale: float = 0.71,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.channels = channels
        self.c_mults = c_mults if c_mults is not None else [1, 2, 4, 8, 16]
        self.strides = strides if strides is not None else [2, 4, 4, 8, 8]
        self.latent_dim = latent_dim
        self.encoder_latent_dim = encoder_latent_dim
        self.use_snake = use_snake
        self.downsample_shortcut = downsample_shortcut
        self.upsample_shortcut = upsample_shortcut
        self.out_shortcut = out_shortcut
        self.in_shortcut = in_shortcut
        self.final_tanh = final_tanh
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.scale = scale


class LongCatAudioDiTConfig(PreTrainedConfig):
    """HF config for the monolithic LongCat-AudioDiT checkpoint."""

    model_type = "audiodit"
    sub_configs = {
        "vae_config": LongCatAudioDiTVaeConfig,
        "text_encoder_config": UMT5Config,
    }

    def __init__(
        self,
        dit_dim: int = 1536,
        dit_depth: int = 24,
        dit_heads: int = 24,
        dit_ff_mult: float = 4.0,
        dit_text_dim: int = 768,
        dit_dropout: float = 0.0,
        dit_bias: bool = True,
        dit_cross_attn: bool = True,
        dit_adaln_type: str = "global",
        dit_adaln_use_text_cond: bool = True,
        dit_long_skip: bool = True,
        dit_text_conv: bool = True,
        dit_qk_norm: bool = True,
        dit_cross_attn_norm: bool = False,
        dit_eps: float = 1e-6,
        dit_use_latent_condition: bool = True,
        repa_dit_layer: int = 8,
        latent_dim: int = 64,
        sigma: float = 0.0,
        sampling_rate: int = 24000,
        latent_hop: int = 2048,
        max_wav_duration: float = 30.0,
        text_encoder_model: str = "google/umt5-base",
        text_add_embed: bool = True,
        text_norm_feat: bool = True,
        vae_config: LongCatAudioDiTVaeConfig | dict | None = None,
        text_encoder_config: UMT5Config | dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dit_dim = dit_dim
        self.dit_depth = dit_depth
        self.dit_heads = dit_heads
        self.dit_ff_mult = dit_ff_mult
        self.dit_text_dim = dit_text_dim
        self.dit_dropout = dit_dropout
        self.dit_bias = dit_bias
        self.dit_cross_attn = dit_cross_attn
        self.dit_adaln_type = dit_adaln_type
        self.dit_adaln_use_text_cond = dit_adaln_use_text_cond
        self.dit_long_skip = dit_long_skip
        self.dit_text_conv = dit_text_conv
        self.dit_qk_norm = dit_qk_norm
        self.dit_cross_attn_norm = dit_cross_attn_norm
        self.dit_eps = dit_eps
        self.dit_use_latent_condition = dit_use_latent_condition
        self.repa_dit_layer = repa_dit_layer
        self.latent_dim = latent_dim
        self.sigma = sigma
        self.sampling_rate = sampling_rate
        self.latent_hop = latent_hop
        self.max_wav_duration = max_wav_duration
        self.text_encoder_model = text_encoder_model
        self.text_add_embed = text_add_embed
        self.text_norm_feat = text_norm_feat

        if isinstance(vae_config, dict):
            vae_config = LongCatAudioDiTVaeConfig(**vae_config)
        self.vae_config = (
            vae_config if vae_config is not None else LongCatAudioDiTVaeConfig()
        )

        if isinstance(text_encoder_config, dict):
            text_encoder_config = UMT5Config(**text_encoder_config)
        self.text_encoder_config = text_encoder_config


@dataclass
class LongCatAudioDiTArchConfig(DiTArchConfig):
    hidden_size: int = 1536
    num_attention_heads: int = 24
    num_channels_latents: int = 64
    attention_head_dim: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attention_head_dim <= 0 and self.num_attention_heads > 0:
            self.attention_head_dim = self.hidden_size // self.num_attention_heads


@dataclass
class LongCatAudioDiTDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LongCatAudioDiTArchConfig)
    prefix: str = "longcat_audiodit"


__all__ = [
    "LongCatAudioDiTConfig",
    "LongCatAudioDiTVaeConfig",
    "LongCatAudioDiTArchConfig",
    "LongCatAudioDiTDitConfig",
]
