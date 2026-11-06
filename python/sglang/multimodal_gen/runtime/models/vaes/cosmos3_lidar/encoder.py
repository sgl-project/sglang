# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 V1.2 LiDAR range-map encoder (inference only).

Behavioral port of the imaginaire4 LiDAR TransformerVAE encoder: a hierarchical
hourglass over per-sweep range images (128 rows x 1808 azimuth columns, the
1800 semantic columns circularly padded by four on each side), with
neighborhood attention for spatial mixing, causal temporal attention between
sweeps, and a joint 3D-attention bottleneck. Only the encoder, the posterior
mean, and the latent affine are needed at inference; the decoder weights that
ship next to them in ``lidar_vae/`` are ignored.

Parameter names mirror the exported ``lidar_vae/diffusion_pytorch_model.safetensors``
so the state dict loads without remapping. Spatial neighborhood attention runs
as a compiled FlexAttention block mask (``neighborhood_attention.py``);
``SGLANG_DIFFUSION_COSMOS3_LIDAR_USE_NATTEN=1`` selects NATTEN's ``na2d`` instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    validate_lidar_config,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar import (
    COSMOS3_LIDAR_VALIDITY_THRESHOLD,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar.layers import (
    LidarBottleneckBlock,
    LidarRMSNorm,
    LidarSpatialBlock,
    LidarSpatialPE,
    LidarTemporalBlock,
    _ChannelsLast,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar.loading import (
    _load_lidar_component,
)

# Range and intensity outside these predicates are encoded as the invalid fill.
COSMOS3_LIDAR_INVALID_VALUE = -1.0
# Causal SDPA batches larger than this overflow the kernel's grid dimension.


def required_lidar_sweeps(
    num_camera_frames: int, camera_fps: float, lidar_fps: float
) -> int:
    """Sweeps captured during ``num_camera_frames`` at ``camera_fps`` (never fewer than one)."""
    if num_camera_frames < 1:
        raise ValueError(
            f"num_camera_frames must be positive, got {num_camera_frames}."
        )
    if camera_fps <= 0 or lidar_fps <= 0:
        raise ValueError(
            f"camera_fps and lidar_fps must be positive, got {camera_fps}, {lidar_fps}."
        )
    return max(1, round(num_camera_frames * lidar_fps / camera_fps))


def pad_lidar_sweeps(frames: torch.Tensor, target: int) -> torch.Tensor:
    """Extend ``[C, T, H, W]`` to ``target`` sweeps by reflection, then last-sweep repeat."""
    if frames.shape[1] >= target:
        return frames[:, :target]
    if frames.shape[1] == 0:
        raise ValueError("Cannot pad an empty LiDAR clip.")
    padded = frames
    while padded.shape[1] < target:
        pad_len = min(padded.shape[1] - 1, target - padded.shape[1])
        if pad_len <= 0:
            repeat = padded[:, -1:].expand(-1, target - padded.shape[1], -1, -1)
            return torch.cat([padded, repeat], dim=1)
        padded = torch.cat([padded, padded.flip(dims=[1])[:, :pad_len]], dim=1)
    return padded


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _SpaceToDepth(nn.Module):
    """``[B, 2H, 2W, C]`` to ``[B, H, W, 4C]`` in (row, column) patch order."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        x = x.view(batch, height // 2, 2, width // 2, 2, channels)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, height // 2, width // 2, 4 * channels
        )


class _ChannelsFirst(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class LidarTransformerEncoder(nn.Module):
    """Hourglass encoder producing posterior parameters ``[B, 2*z_dim, T, H_z, W_z]``."""

    def __init__(self, network: Mapping[str, Any]) -> None:
        super().__init__()
        resolution = tuple(int(v) for v in network["resolution"])
        patch = tuple(int(v) for v in network["patch_size"])
        depths = [int(v) for v in network["depths"]]
        heads = [int(v) for v in network["num_heads"]]
        dilations = [int(v) for v in network.get("dilation", [1] * len(depths))]
        window = tuple(int(v) for v in network["window_size"])
        base = int(network.get("base_channels", 128))
        z_dim = int(network["z_dim"])
        mlp_ratio = float(network.get("mlp_ratio", 3.0))
        circular = bool(network.get("circular_padding", True))
        if any(network.get("temporal_downsample", [])):
            raise NotImplementedError(
                "Cosmos3 LiDAR V1.2 encoding assumes no temporal downsampling."
            )
        if not network.get("bottleneck_3d", False) or not network.get(
            "bottleneck_3d_rope", False
        ):
            raise NotImplementedError(
                "Cosmos3 LiDAR V1.2 encoding assumes the 3D-RoPE joint bottleneck."
            )
        if network.get("temporal_mixer", "attention") != "attention":
            raise NotImplementedError(
                "Cosmos3 LiDAR V1.2 encoding assumes temporal attention."
            )
        self.patch = patch
        self.depths = depths
        token_size = (resolution[0] // patch[0], resolution[1] // patch[1])
        max_harmonics = (token_size[0] // 2, token_size[1] // 2)
        self.tokenizer = nn.Sequential(
            nn.Conv2d(
                int(network["in_channels"]),
                base,
                kernel_size=patch,
                stride=patch,
                bias=False,
            ),
            _ChannelsLast(),
        )
        self.spatial_pe = LidarSpatialPE(base, token_size)
        self.down_levels = nn.ModuleDict()
        for level, num_blocks in enumerate(depths[:-1]):
            dim = base << level
            harmonics = (
                max(max_harmonics[0] >> level, 1),
                max(max_harmonics[1] >> level, 1),
            )
            self.down_levels[f"spatial_{level}"] = nn.ModuleList(
                [
                    LidarSpatialBlock(
                        dim,
                        heads[level],
                        window,
                        (1, 1) if j % 2 == 0 else (dilations[level], dilations[level]),
                        harmonics,
                        mlp_ratio,
                        circular,
                    )
                    for j in range(num_blocks)
                ]
            )
            if level > 0:
                self.down_levels[f"temporal_{level}"] = nn.ModuleList(
                    [
                        LidarTemporalBlock(dim, heads[level], mlp_ratio)
                        for _ in range(num_blocks)
                    ]
                )
            self.down_levels[f"merge_{level}"] = nn.Sequential(
                _SpaceToDepth(), nn.Linear(4 * dim, 2 * dim, bias=False)
            )
        level = len(depths) - 1
        bottleneck_dim = base << level
        bottleneck_size = (token_size[0] >> level, token_size[1] >> level)
        self.mid_3d = nn.ModuleList(
            [
                LidarBottleneckBlock(
                    bottleneck_dim,
                    heads[-1],
                    int(network.get("bottleneck_3d_max_t", 32)),
                    bottleneck_size[0],
                    bottleneck_size[1],
                    mlp_ratio,
                )
                for _ in range(depths[-1])
            ]
        )
        self.head = nn.Sequential(
            LidarRMSNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, z_dim * 2, bias=False),
            _ChannelsFirst(),
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Encode ``[B, C, T, H, W]`` sweeps attending to cached earlier sweeps."""
        batch, _, frames, _, _ = x.shape
        old_cache = dict(kv_cache or {})
        new_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        coords = F.avg_pool2d(coords, kernel_size=self.patch, stride=self.patch)
        h = self.tokenizer(x.permute(0, 2, 1, 3, 4).flatten(0, 1)) + self.spatial_pe()
        _, height, width, channels = h.shape
        h = h.view(batch, frames, height, width, channels)
        for level in range(len(self.depths) - 1):
            temporal_key = f"temporal_{level}"
            temporal = (
                self.down_levels[temporal_key]
                if temporal_key in self.down_levels
                else None
            )
            for index, spatial_block in enumerate(self.down_levels[f"spatial_{level}"]):
                _, _, height, width, channels = h.shape
                h = spatial_block(h.flatten(0, 1), coords)
                if temporal is not None:
                    h = h.view(batch, frames, height, width, channels).permute(
                        0, 2, 3, 1, 4
                    )
                    h = h.reshape(batch * height * width, frames, channels)
                    key = f"down_levels.temporal_{level}.{index}"
                    h, new_cache[key] = temporal[index].forward_stream(
                        h, old_cache.pop(key, None)
                    )
                    h = h.view(batch, height, width, frames, channels).permute(
                        0, 3, 1, 2, 4
                    )
                else:
                    h = h.view(batch, frames, height, width, channels)
            h = self.down_levels[f"merge_{level}"](h.flatten(0, 1))
            h = h.view(batch, frames, *h.shape[1:])
            coords = F.avg_pool2d(coords, kernel_size=2, stride=2)
        for index, block in enumerate(self.mid_3d):
            key = f"mid_3d.{index}"
            h, new_cache[key] = block.forward_stream(h, old_cache.pop(key, None))
        _, _, height, width, channels = h.shape
        h = self.head(h.reshape(batch * frames, height, width, channels))
        h = h.view(batch, frames, *h.shape[1:]).permute(0, 2, 1, 3, 4)
        if old_cache:
            raise ValueError(
                f"Unused LiDAR temporal cache entries: {sorted(old_cache)}."
            )
        return h, new_cache


class Cosmos3LidarEncoder(nn.Module):
    """Metric range maps in, normalized LiDAR latents out (posterior mean, FP32)."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = validate_lidar_config(config)
        network = self.config["network_config"]
        channels = int(self.config["latent_channels"])
        self.encoder = LidarTransformerEncoder(network)
        self.quant_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=1)
        height, width = (int(v) for v in network["resolution"])
        self.register_buffer("coords", torch.zeros(1, 2, height, width))
        self.register_buffer("latent_mean", torch.zeros(1, channels, 1, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, channels, 1, 1, 1))

    @property
    def projection(self) -> Mapping[str, Any]:
        return self.config["range_projection"]

    @property
    def fps(self) -> float:
        return float(self.config["fps"])

    @property
    def temporal_compression_factor(self) -> int:
        return int(self.config["temporal_compression_factor"])

    @classmethod
    def from_pretrained(
        cls, model_path: str, config: Mapping[str, Any], device: torch.device | str
    ) -> Cosmos3LidarEncoder:
        return _load_lidar_component(
            cls, model_path, config, device, ("encoder.", "quant_conv.")
        )

    def prepare_input(self, frames: torch.Tensor) -> torch.Tensor:
        """Metric ``[3, T, 128, W]`` range maps to the network's ``[1, 3, T, 128, 1808]``."""
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        if frames.ndim != 5 or frames.shape[1] != 3:
            raise ValueError(
                f"LiDAR frames must be [3, T, H, W], got {tuple(frames.shape)}."
            )
        frames = frames.to(device=self.coords.device, dtype=torch.float32)
        projection = self.projection
        semantic, model_width = (
            int(projection["semantic_width"]),
            int(projection["model_width"]),
        )
        width = frames.shape[-1]
        if width == semantic:
            half = (model_width - semantic) // 2
            frames = torch.cat(
                (frames[..., -half:], frames, frames[..., :half]), dim=-1
            )
        elif width != model_width:
            raise ValueError(
                f"LiDAR frames must be {semantic} (semantic) or {model_width} (model) columns wide, got {width}."
            )
        if frames.shape[-2] != int(projection["native_height"]):
            raise ValueError(
                f"LiDAR frames must have {projection['native_height']} rows, got {frames.shape[-2]}."
            )
        minimum, maximum = (
            float(projection["min_range_m"]),
            float(projection["max_range_m"]),
        )
        ranges, intensities, validity = frames.split(1, dim=1)
        valid = (
            (validity >= COSMOS3_LIDAR_VALIDITY_THRESHOLD)
            & (ranges >= minimum)
            & (ranges <= maximum)
        )
        ranges = (ranges.clamp(minimum, maximum) - minimum) / (
            maximum - minimum
        ) * 2.0 - 1.0
        intensities = intensities.clamp(0.0, 1.0) * 2.0 - 1.0
        fill = COSMOS3_LIDAR_INVALID_VALUE
        return torch.cat(
            (
                ranges.masked_fill(~valid, fill),
                intensities.masked_fill(~valid, fill),
                valid.float(),
            ),
            dim=1,
        )

    @torch.inference_mode()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode metric range maps to normalized latents ``[1, C, T, H_z, W_z]``."""
        with torch.autocast(device_type=self.coords.device.type, enabled=False):
            state = self.prepare_input(frames)
            chunk = int(self.config["streaming_chunk_frames"])
            context = self.config["streaming_context_frames"]
            outputs = []
            cache = None
            for start in range(0, state.shape[2], chunk):
                pixels = state[:, :, start : start + chunk]
                if cache is not None and context is not None:
                    keep = int(context) - pixels.shape[2]
                    cache = (
                        {
                            key: (k[:, :, -keep:], v[:, :, -keep:])
                            for key, (k, v) in cache.items()
                        }
                        if keep > 0
                        else None
                    )
                parameters, cache = self.encoder.forward_stream(
                    pixels, self.coords, cache
                )
                batch, _, frames_out, height, width = parameters.shape
                flat = parameters.permute(0, 2, 1, 3, 4).flatten(0, 1)
                mean = self.quant_conv(flat).chunk(2, dim=1)[0]
                outputs.append(
                    mean.view(batch, frames_out, -1, height, width).permute(
                        0, 2, 1, 3, 4
                    )
                )
            latent = torch.cat(outputs, dim=2)
            return (latent - self.latent_mean) / self.latent_std
