# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import argparse
import dataclasses
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.base import ArchConfig, ModelConfig
from sglang.multimodal_gen.utils import StoreBoolean

AUTO_PARALLEL_DECODE_MODE = "auto"
SPATIAL_SHARD_PARALLEL_DECODE_MODES = ("spatial_shard", "spatial")


@lru_cache(maxsize=8)
def is_spatial_shard_parallel_decode_mode(mode: str) -> bool:
    return mode in SPATIAL_SHARD_PARALLEL_DECODE_MODES


@lru_cache(maxsize=8)
def is_auto_parallel_decode_mode(mode: str) -> bool:
    return mode == AUTO_PARALLEL_DECODE_MODE


@lru_cache(maxsize=128)
def _should_use_auto_spatial_shard_parallel_decode(
    z_shape: tuple[int, ...],
    world_size: int,
    min_latent_elements_per_rank: int,
) -> bool:
    if world_size <= 1 or z_shape[-2] < world_size:
        return False
    latent_elements_per_rank = (
        z_shape[0] * z_shape[-3] * z_shape[-2] * z_shape[-1]
    ) // world_size
    return latent_elements_per_rank >= min_latent_elements_per_rank


def should_use_spatial_shard_parallel_decode(
    config: Any, z: torch.Tensor | None = None, world_size: int = 1
) -> bool:
    if not config.use_parallel_decode:
        return False

    if is_spatial_shard_parallel_decode_mode(config.parallel_decode_mode):
        return True

    if not is_auto_parallel_decode_mode(config.parallel_decode_mode):
        return False

    if not config.auto_parallel_decode_prefers_spatial_shard():
        return False

    if z is None:
        return True

    return config.should_use_auto_spatial_shard_parallel_decode(z, world_size)


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
    use_parallel_decode: bool = True
    parallel_decode_mode: str = AUTO_PARALLEL_DECODE_MODE
    auto_parallel_decode_min_latent_elements_per_rank: int = 4096

    def __post_init__(self):
        self.blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        )

    def post_init(self):
        pass

    def auto_parallel_decode_prefers_spatial_shard(self) -> bool:
        return False

    def should_use_auto_spatial_shard_parallel_decode(
        self, z: torch.Tensor, world_size: int
    ) -> bool:
        return _should_use_auto_spatial_shard_parallel_decode(
            tuple(z.shape),
            world_size,
            self.auto_parallel_decode_min_latent_elements_per_rank,
        )

    @staticmethod
    def add_cli_args(parser: Any, prefix: str = "vae-config") -> Any:
        """Add CLI arguments for VAEConfig fields"""
        parser.add_argument(
            f"--{prefix}.load-encoder",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.load_encoder",
            default=None,
            help="Whether to load the VAE encoder",
        )
        parser.add_argument(
            f"--{prefix}.load-decoder",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.load_decoder",
            default=None,
            help="Whether to load the VAE decoder",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-min-height",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_min_height",
            default=None,
            help="Minimum height for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-min-width",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_min_width",
            default=None,
            help="Minimum width for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-min-num-frames",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_min_num_frames",
            default=None,
            help="Minimum number of frames for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-stride-height",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_stride_height",
            default=None,
            help="Stride height for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-stride-width",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_stride_width",
            default=None,
            help="Stride width for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.tile-sample-stride-num-frames",
            type=int,
            dest=f"{prefix.replace('-', '_')}.tile_sample_stride_num_frames",
            default=None,
            help="Stride number of frames for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.blend-num-frames",
            type=int,
            dest=f"{prefix.replace('-', '_')}.blend_num_frames",
            default=None,
            help="Number of frames to blend for VAE tile sampling",
        )
        parser.add_argument(
            f"--{prefix}.use-tiling",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.use_tiling",
            default=None,
            help="Whether to use tiling for VAE",
        )
        parser.add_argument(
            f"--{prefix}.use-temporal-tiling",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.use_temporal_tiling",
            default=None,
            help="Whether to use temporal tiling for VAE",
        )
        parser.add_argument(
            f"--{prefix}.use-parallel-tiling",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.use_parallel_tiling",
            default=None,
            help="Whether to use parallel tiling for VAE",
        )
        parser.add_argument(
            f"--{prefix}.use-parallel-decode",
            action=StoreBoolean,
            dest=f"{prefix.replace('-', '_')}.use_parallel_decode",
            default=None,
            help="Whether to use parallel decode for VAE",
        )
        parser.add_argument(
            f"--{prefix}.parallel-decode-mode",
            choices=("tiled", "patch", "spatial_shard", "spatial", "auto"),
            dest=f"{prefix.replace('-', '_')}.parallel_decode_mode",
            default=None,
            help="Parallel decode mode for VAE",
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
