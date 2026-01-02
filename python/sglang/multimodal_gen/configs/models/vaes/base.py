# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.base import ArchConfig, ModelConfig
from sglang.multimodal_gen.utils import StoreBoolean


@dataclass
class VAEArchConfig(ArchConfig):
    scaling_factor: float | torch.Tensor = 0

    temporal_compression_ratio: int = 4
    # or vae_scale_factor?
    spatial_compression_ratio: int = 8


@dataclass
class VAEConfig(ModelConfig):
    arch_config: VAEArchConfig = field(default_factory=VAEArchConfig)

    # sglang-diffusion VAE-specific parameters
    load_encoder: bool = True
    load_decoder: bool = True

    tile_sample_min_height: int = 256
    tile_sample_min_width: int = 256
    tile_sample_min_num_frames: int = 16
    tile_sample_stride_height: int = 192
    tile_sample_stride_width: int = 192
    tile_sample_stride_num_frames: int = 12
    blend_num_frames: int = 0

    use_tiling: bool = True
    use_temporal_tiling: bool = True
    use_parallel_tiling: bool = True
    use_temporal_scaling_frames: bool = True

    def __post_init__(self):
        self.blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        )

    def post_init(self):
        pass

    @staticmethod
    def add_cli_args(parser: Any, prefix: str = "vae-config") -> Any:
        """Add CLI arguments for VAEConfig fields"""
        parser.add_argument(
            f"--{prefix}.load-encoder",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.load_encoder",
            default=VAEConfig.load_encoder,
            help="Whether to load the VAE encoder",
        )
        parser.add_argument(
            f"--{prefix}.load-decoder",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.load_decoder",
            default=VAEConfig.load_decoder,
            help="Whether to load the VAE decoder",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-min-height",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_min_height",
            default=VAEConfig.tile_sample_min_height,
            help="Minimum height for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-min-width",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_min_width",
            default=VAEConfig.tile_sample_min_width,
            help="Minimum width for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-min-num-frames",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_min_num_frames",
            default=VAEConfig.tile_sample_min_num_frames,
            help="Minimum number of frames for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-stride-height",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_stride_height",
            default=VAEConfig.tile_sample_stride_height,
            help="Stride height for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-stride-width",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_stride_width",
            default=VAEConfig.tile_sample_stride_width,
            help="Stride width for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-stride-num-frames",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_stride_num_frames",
            default=VAEConfig.tile_sample_stride_num_frames,
            help="Stride number of frames for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.blend-num-frames",
            type=int,
            dest=f"{prefix.replace('-', '_')}.blend_num_frames",
            default=VAEConfig.blend_num_frames,
            help="Number of frames to blend for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.use-tiling",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.use_tiling",
            default=VAEConfig.use_tiling,
            help="Whether to use tiling for VAE",
        )
        parser.add_argument(
            f"--{prefix}.use-temporal-tiling",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.use_temporal_tiling",
            default=VAEConfig.use_temporal_tiling,
            help="Whether to use temporal tiling for VAE",
        )
        parser.add_argument(
            f"--{prefix}.use-parallel-tiling",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.use_parallel_tiling",
            default=VAEConfig.use_parallel_tiling,
            help="Whether to use parallel tiling for VAE",
        )

        return parser

    def get_vae_scale_factor(self):
        return 2 ** (len(self.arch_config.block_out_channels) - 1)

    def encode_sample_mode(self):
        return "argmax"

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "VAEConfig":
        kwargs = {}
        for attr in dataclasses.fields(cls):
            value = getattr(args, attr.name, None)
            if value is not None:
                kwargs[attr.name] = value
        return cls(**kwargs)
