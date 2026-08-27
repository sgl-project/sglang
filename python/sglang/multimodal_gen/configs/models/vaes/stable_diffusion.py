# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion AutoencoderKL configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class StableDiffusionVAEArchConfig(VAEArchConfig):
    scaling_factor: float = 0.18215
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    sample_size: int = 32
    block_out_channels: tuple[int, ...] = (64,)
    layers_per_block: int = 1
    act_fn: str = "silu"
    norm_num_groups: int = 32
    down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",)
    up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",)
    mid_block_add_attention: bool = True
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    force_upcast: bool = True


@dataclass
class StableDiffusionVAEConfig(VAEConfig):
    arch_config: StableDiffusionVAEArchConfig = field(
        default_factory=StableDiffusionVAEArchConfig
    )
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    use_parallel_decode: bool = False
    use_temporal_scaling_frames: bool = False
