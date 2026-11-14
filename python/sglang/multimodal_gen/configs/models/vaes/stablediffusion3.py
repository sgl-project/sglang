# SPDX-License-Identifier: Apache-2.0
"""StableDiffusion3 VAE configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class StableDiffusion3VAEArchConfig(VAEArchConfig):
    """Architecture configuration for StableDiffusion3 VAE."""

    scaling_factor: float = 1.5305
    shift_factor: float = 0.0609

    spatial_compression_ratio: int = 8
    temporal_compression_ratio: int = 1

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    sample_size: int = 128

    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    norm_num_groups: int = 32

    down_block_types: tuple[str, ...] = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    )
    up_block_types: tuple[str, ...] = (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )

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

    tile_sample_min_height: int = 512
    tile_sample_min_width: int = 512
    tile_sample_min_num_frames: int = 1
    tile_sample_stride_height: int = 448
    tile_sample_stride_width: int = 448
    tile_sample_stride_num_frames: int = 1

    use_tiling: bool = True
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = True
    use_temporal_scaling_frames: bool = False

    def __post_init__(self) -> None:
        """Post initialization for SD3 VAE specific setup."""
        super().__post_init__()
        self.update_model_arch({"_class_name": "AutoencoderKL"})
        self.blend_num_frames = 0
