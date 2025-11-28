# SPDX-License-Identifier: Apache-2.0
"""StableDiffusion3 VAE configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class StableDiffusion3VAEArchConfig(VAEArchConfig):
    """Architecture configuration for StableDiffusion3 VAE."""

    # SD3 VAE specific parameters
    scaling_factor: float = 1.5305  # SD3 specific scaling factor
    shift_factor: float = 0.0609  # SD3 specific shift factor

    # Compression ratios
    spatial_compression_ratio: int = 8  # 8x spatial compression
    temporal_compression_ratio: int = 1  # No temporal compression

    # Model architecture parameters
    in_channels: int = 3  # RGB input
    out_channels: int = 3  # RGB output
    latent_channels: int = 16  # SD3 transformer input channels
    sample_size: int = 128  # Standard input size

    # Encoder/Decoder block configurations
    block_out_channels: tuple = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    norm_num_groups: int = 32

    # Block types for encoder and decoder
    down_block_types: tuple = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    )
    up_block_types: tuple = (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )

    # Attention configuration
    attention_head_dim: int = 8
    mid_block_add_attention: bool = True
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False


@dataclass
class StableDiffusion3VAEConfig(VAEConfig):
    """Configuration for StableDiffusion3 VAE."""

    arch_config: StableDiffusion3VAEArchConfig = field(
        default_factory=StableDiffusion3VAEArchConfig
    )

    # Override base VAE parameters for SD3
    tile_sample_min_height: int = 512
    tile_sample_min_width: int = 512
    tile_sample_min_num_frames: int = 1  # Image VAE
    tile_sample_stride_height: int = 448
    tile_sample_stride_width: int = 448
    tile_sample_stride_num_frames: int = 1

    # Tiling configuration optimized for SD3
    use_tiling: bool = True
    use_temporal_tiling: bool = False  # Disable for image VAE
    use_parallel_tiling: bool = True
    use_temporal_scaling_frames: bool = False

    def __post_init__(self):
        """Post initialization for SD3 VAE specific setup."""
        super().__post_init__()

        # Set model class name for SD3 VAE
        self.update_model_arch({"_class_name": "AutoencoderKL"})

        # Update blend frames for image VAE
        self.blend_num_frames = 0

    def post_init(self):
        """Additional initialization after model loading."""
        pass

    @classmethod
    def from_pretrained_config(cls, config_dict: dict) -> "StableDiffusion3VAEConfig":
        """Create config from diffusers pretrained model config."""
        arch_config = StableDiffusion3VAEArchConfig()

        # Update from config dict if provided
        if config_dict:
            arch_config.scaling_factor = config_dict.get("scaling_factor", 1.5305)
            arch_config.shift_factor = config_dict.get("shift_factor", 0.0609)
            arch_config.in_channels = config_dict.get("in_channels", 3)
            arch_config.out_channels = config_dict.get("out_channels", 3)
            arch_config.latent_channels = config_dict.get("latent_channels", 16)
            arch_config.sample_size = config_dict.get("sample_size", 128)
            arch_config.block_out_channels = tuple(
                config_dict.get("block_out_channels", [128, 256, 512, 512])
            )
            arch_config.layers_per_block = config_dict.get("layers_per_block", 2)
            arch_config.norm_num_groups = config_dict.get("norm_num_groups", 32)
            arch_config.attention_head_dim = config_dict.get("attention_head_dim", 8)

        return cls(arch_config=arch_config)

    def get_vae_scale_factor(self) -> int:
        """Calculate VAE scale factor for image processing."""
        return 2 ** (len(self.arch_config.block_out_channels) - 1)

    def get_latent_shape(self, height: int, width: int) -> tuple:
        """Calculate latent shape from input dimensions."""
        scale_factor = self.get_vae_scale_factor()
        latent_height = height // scale_factor
        latent_width = width // scale_factor
        return (
            self.arch_config.latent_channels,
            latent_height,
            latent_width,
        )
