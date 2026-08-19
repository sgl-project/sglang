# Copied and adapted from: https://github.com/meituan-longcat/LongCat-AudioDiT
"""LongCat-AudioDiT model configuration"""

from transformers import PreTrainedConfig, logging
from transformers.models.umt5.configuration_umt5 import UMT5Config

logger = logging.get_logger(__name__)


class LongCatAudioDiTVaeConfig(PreTrainedConfig):
    r"""
    Configuration class for the LongCat-AudioDiT WAV-VAE audio autoencoder.

    Args:
        in_channels (`int`, *optional*, defaults to 1):
            Number of input audio channels (mono=1).
        channels (`int`, *optional*, defaults to 128):
            Base channel count for encoder/decoder.
        c_mults (`list[int]`, *optional*, defaults to `[1, 2, 4, 8, 16]`):
            Channel multipliers for each encoder/decoder stage.
        strides (`list[int]`, *optional*, defaults to `[2, 4, 4, 8, 8]`):
            Downsampling strides for each encoder stage.
        latent_dim (`int`, *optional*, defaults to 64):
            Dimensionality of the latent space (after VAE bottleneck: encoder outputs 128, split to mean+scale → 64).
        encoder_latent_dim (`int`, *optional*, defaults to 128):
            Dimensionality of the encoder output before VAE bottleneck.
        use_snake (`bool`, *optional*, defaults to `True`):
            Whether to use Snake activation instead of ELU.
        downsample_shortcut (`str`, *optional*, defaults to `"averaging"`):
            Shortcut type for encoder downsampling blocks.
        upsample_shortcut (`str`, *optional*, defaults to `"duplicating"`):
            Shortcut type for decoder upsampling blocks.
        out_shortcut (`str`, *optional*, defaults to `"averaging"`):
            Shortcut type for encoder output projection.
        in_shortcut (`str`, *optional*, defaults to `"duplicating"`):
            Shortcut type for decoder input projection.
        final_tanh (`bool`, *optional*, defaults to `False`):
            Whether to apply tanh to the decoder output.
        downsampling_ratio (`int`, *optional*, defaults to 2048):
            Total downsampling ratio from audio samples to latent frames.
        sample_rate (`int`, *optional*, defaults to 24000):
            Audio sample rate.
        scale (`float`, *optional*, defaults to 0.71):
            Scale factor for the latent space.
    """

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
    r"""
    Configuration class for LongCat-AudioDiT, a Conditional Flow Matching TTS model based on DiT architecture.

    Args:
        dit_dim (`int`, *optional*, defaults to 1536):
            Hidden dimension of the DiT transformer.
        dit_depth (`int`, *optional*, defaults to 24):
            Number of transformer layers.
        dit_heads (`int`, *optional*, defaults to 24):
            Number of attention heads.
        dit_ff_mult (`float`, *optional*, defaults to 4.0):
            Feed-forward network multiplier.
        dit_text_dim (`int`, *optional*, defaults to 768):
            Dimension of the text encoder output (UMT5-base).
        dit_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate.
        dit_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in linear layers.
        dit_cross_attn (`bool`, *optional*, defaults to `True`):
            Whether to use cross-attention layers.
        dit_adaln_type (`str`, *optional*, defaults to `"global"`):
            Type of adaptive layer norm (`"global"` or `"local"`).
        dit_adaln_use_text_cond (`bool`, *optional*, defaults to `True`):
            Whether to condition AdaLN on text embeddings.
        dit_long_skip (`bool`, *optional*, defaults to `True`):
            Whether to use long skip connection (input added to output).
        dit_text_conv (`bool`, *optional*, defaults to `True`):
            Whether to apply ConvNeXt blocks on text embeddings.
        dit_qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply RMS normalization to Q and K.
        dit_cross_attn_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization in cross-attention.
        dit_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for normalization layers.
        dit_use_latent_condition (`bool`, *optional*, defaults to `True`):
            Whether to use latent conditioning (for prompt audio).
        repa_dit_layer (`int`, *optional*, defaults to 8):
            Layer index for representation alignment.
        latent_dim (`int`, *optional*, defaults to 64):
            Dimensionality of the audio latent space.
        sigma (`float`, *optional*, defaults to 0.0):
            Noise level for conditional flow matching.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Audio sample rate.
        latent_hop (`int`, *optional*, defaults to 2048):
            Hop size in audio samples per latent frame.
        max_wav_duration (`float`, *optional*, defaults to 30.0):
            Maximum audio duration in seconds.
        text_encoder_model (`str`, *optional*, defaults to `"google/umt5-base"`):
            HuggingFace model identifier for the text encoder.
        text_add_embed (`bool`, *optional*, defaults to `True`):
            Whether to add the first hidden state to the last hidden state in text encoding.
        text_norm_feat (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization to text features.
        vae_config (`LongCatAudioDiTVaeConfig` or `dict`, *optional*):
            Configuration for the WAV-VAE audio autoencoder.
    """

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


__all__ = ["LongCatAudioDiTConfig", "LongCatAudioDiTVaeConfig"]
