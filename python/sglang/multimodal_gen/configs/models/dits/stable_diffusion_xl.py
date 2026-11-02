# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion XL UNet configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class StableDiffusionXLUNetArchConfig(DiTArchConfig):
    """Defaults match stabilityai/stable-diffusion-xl-base-1.0; the loader overwrites
    them from the checkpoint's unet/config.json."""

    in_channels: int = 4
    out_channels: int = 4
    sample_size: int = 128
    block_out_channels: tuple[int, ...] = (320, 640, 1280)
    # unet/config.json's `attention_head_dim` holds per-level head counts, while the
    # runtime reads `attention_head_dim` as the head size; __post_init__ splits them.
    attention_head_dim: int = 64
    num_attention_heads_per_level: tuple[int, ...] = (5, 10, 20)
    cross_attention_dim: int = 2048
    addition_time_embed_dim: int = 256

    # The native module nests the UNet under `unet.`; Diffusers checkpoints do not.
    param_names_mapping: dict = field(
        default_factory=lambda: {r"^(?!unet\.)(.*)$": r"unet.\1"}
    )
    reverse_param_names_mapping: dict = field(
        default_factory=lambda: {r"^unet\.(.*)$": r"\1"}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.attention_head_dim, int):
            self.num_attention_heads_per_level = tuple(self.attention_head_dim)
        self.hidden_size = self.block_out_channels[-1]
        self.num_attention_heads = self.num_attention_heads_per_level[-1]
        self.attention_head_dim = self.hidden_size // self.num_attention_heads
        self.num_channels_latents = self.in_channels


@dataclass
class StableDiffusionXLUNetConfig(DiTConfig):
    arch_config: StableDiffusionXLUNetArchConfig = field(
        default_factory=StableDiffusionXLUNetArchConfig
    )
    prefix: str = "SDXL"
