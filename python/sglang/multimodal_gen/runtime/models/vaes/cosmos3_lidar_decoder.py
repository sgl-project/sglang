# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 V1.2 LiDAR range-map decoder (inference only).

Mirror image of ``cosmos3_lidar_encoder``: a stem at the bottleneck grid, the
joint 3D-RoPE bottleneck blocks, three up levels of temporal-then-neighborhood
attention with 2x patch expanding, and a detokenizer back to three channels of
network-space range, intensity, and mask logit. Parameter names follow the
exported ``lidar_vae/diffusion_pytorch_model.safetensors`` (``decoder.*``,
``post_quant_conv.*``). Latents arrive normalized by the shared latent
statistics; output is the metric clip ``[B, 3, T, 128, 1800]`` of range in
meters, unit intensity, and a resolved ``{0, 1}`` validity mask.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    validate_lidar_config,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar_encoder import (
    COSMOS3_LIDAR_COMPONENT,
    COSMOS3_LIDAR_VALIDITY_THRESHOLD,
    COSMOS3_LIDAR_WEIGHTS_FILE,
    LidarBottleneckBlock,
    LidarRMSNorm,
    LidarSpatialBlock,
    LidarSpatialPE,
    LidarTemporalBlock,
    _ChannelsLast,
    _check_component_matches,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def depth_to_space(x: torch.Tensor) -> torch.Tensor:
    """``[B, H, W, 4C]`` in (row, column, channel) group order to ``[B, 2H, 2W, C]``.

    Inverse of the encoder's ``_SpaceToDepth``; the checkpoint was trained with
    einops' ``(P1 P2 C)`` grouping, so the order is load-bearing.
    """
    batch, height, width, channels = x.shape
    x = x.view(batch, height, width, 2, 2, channels // 4)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(
        batch, height * 2, width * 2, channels // 4
    )


class _PatchExpanding(nn.Module):
    """``[B, H, W, C]`` to ``[B, 2H, 2W, C // 2]``; inverse of the encoder's merge."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depth_to_space(self.linear(x))


class _DepthToPixels(nn.Module):
    """``[B, H, W, p*p*C]`` to ``[B, C, H*p, W*p]`` in (row, column, channel) order."""

    def __init__(self, patch: tuple[int, int], channels: int) -> None:
        super().__init__()
        self.patch = patch
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = x.shape
        p_h, p_w = self.patch
        x = x.view(batch, height, width, p_h, p_w, self.channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, self.channels, height * p_h, width * p_w
        )


class LidarTransformerDecoder(nn.Module):
    """Hourglass decoder from ``[B, z_dim, T, H_z, W_z]`` to network-space ``[B, 3, T, H, W]``."""

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
        out_channels = int(network.get("out_channels", 3))
        mlp_ratio = float(network.get("mlp_ratio", 3.0))
        circular = bool(network.get("circular_padding", True))
        if any(network.get("temporal_upsample") or []) or any(
            network.get("temporal_downsample", [])
        ):
            raise NotImplementedError(
                "Cosmos3 LiDAR V1.2 decoding assumes no temporal resampling."
            )
        if not network.get("bottleneck_3d", False) or not network.get(
            "bottleneck_3d_rope", False
        ):
            raise NotImplementedError(
                "Cosmos3 LiDAR V1.2 decoding assumes the 3D-RoPE joint bottleneck."
            )
        if network.get("temporal_mixer", "attention") != "attention":
            raise NotImplementedError(
                "Cosmos3 LiDAR V1.2 decoding assumes temporal attention."
            )
        for key in (
            "decoder_depths",
            "decoder_num_heads",
            "decoder_dilation",
            "out_patch_size",
        ):
            if network.get(key) is not None:
                raise NotImplementedError(
                    f"Cosmos3 LiDAR V1.2 decoding assumes network_config.{key} is null."
                )
        if network.get("predict_validity", False) or out_channels != 3:
            raise NotImplementedError(
                "Cosmos3 LiDAR V1.2 decoding assumes three output channels with the "
                "mask logit in channel 2."
            )
        self.patch = patch
        self.depths = depths
        self.out_channels = out_channels
        token_size = (resolution[0] // patch[0], resolution[1] // patch[1])
        max_harmonics = (token_size[0] // 2, token_size[1] // 2)
        n_down = len(depths) - 1
        bottleneck_dim = base << n_down
        bottleneck_size = (token_size[0] >> n_down, token_size[1] >> n_down)
        # Index 1 keeps the exported ``stem.1.weight`` name; index 0 only permutes.
        self.stem = nn.Sequential(
            _ChannelsLast(), nn.Linear(z_dim, bottleneck_dim, bias=False)
        )
        self.spatial_pe = LidarSpatialPE(bottleneck_dim, bottleneck_size)
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
        self.up_levels = nn.ModuleDict()
        for level in reversed(range(n_down)):
            dim = base << level
            harmonics = (
                max(max_harmonics[0] >> level, 1),
                max(max_harmonics[1] >> level, 1),
            )
            self.up_levels[f"expand_{level}"] = _PatchExpanding(base << (level + 1))
            self.up_levels[f"spatial_{level}"] = nn.ModuleList(
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
                    for j in range(depths[level])
                ]
            )
            # The top level has no temporal mixing, as in the encoder.
            if level > 0:
                self.up_levels[f"temporal_{level}"] = nn.ModuleList(
                    [
                        LidarTemporalBlock(dim, heads[level], mlp_ratio)
                        for _ in range(depths[level])
                    ]
                )
        self.detokenizer = nn.Sequential(
            LidarRMSNorm(base),
            nn.Linear(base, out_channels * patch[0] * patch[1], bias=False),
            _DepthToPixels(patch, out_channels),
        )

    def forward_stream(
        self,
        z: torch.Tensor,
        coords: torch.Tensor,
        kv_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Decode ``[B, z_dim, T, H_z, W_z]`` sweeps attending to cached earlier sweeps."""
        batch, _, frames, _, _ = z.shape
        old_cache = dict(kv_cache or {})
        new_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        n_down = len(self.depths) - 1
        coord_levels = [F.avg_pool2d(coords, kernel_size=self.patch, stride=self.patch)]
        for _ in range(n_down):
            coord_levels.append(F.avg_pool2d(coord_levels[-1], kernel_size=2, stride=2))
        h = self.stem(z.permute(0, 2, 1, 3, 4).flatten(0, 1)) + self.spatial_pe()
        h = h.view(batch, frames, *h.shape[1:])
        for index, block in enumerate(self.mid_3d):
            key = f"mid_3d.{index}"
            h, new_cache[key] = block.forward_stream(h, old_cache.pop(key, None))
        for level in reversed(range(n_down)):
            h = self.up_levels[f"expand_{level}"](h.flatten(0, 1))
            h = h.view(batch, frames, *h.shape[1:])
            _, _, height, width, channels = h.shape
            temporal_key = f"temporal_{level}"
            temporal = (
                self.up_levels[temporal_key] if temporal_key in self.up_levels else None
            )
            for index, spatial_block in enumerate(self.up_levels[f"spatial_{level}"]):
                if temporal is not None:
                    h = h.permute(0, 2, 3, 1, 4).reshape(
                        batch * height * width, frames, channels
                    )
                    key = f"up_levels.temporal_{level}.{index}"
                    h, new_cache[key] = temporal[index].forward_stream(
                        h, old_cache.pop(key, None)
                    )
                    h = h.view(batch, height, width, frames, channels).permute(
                        0, 3, 1, 2, 4
                    )
                h = spatial_block(h.flatten(0, 1), coord_levels[level])
                h = h.view(batch, frames, height, width, channels)
        h = self.detokenizer(h.flatten(0, 1))
        h = h.view(batch, frames, *h.shape[1:]).permute(0, 2, 1, 3, 4)
        if old_cache:
            raise ValueError(
                f"Unused LiDAR temporal cache entries: {sorted(old_cache)}."
            )
        return h, new_cache


def lidar_network_to_metric(
    network: torch.Tensor,
    *,
    min_range_m: float,
    max_range_m: float,
    validity_threshold: float = COSMOS3_LIDAR_VALIDITY_THRESHOLD,
    apply_validity_mask: bool = True,
) -> torch.Tensor:
    """Network-space ``[B, 3, T, H, W]`` to metric range, unit intensity, and validity.

    Channel 2 of the decoder output is a mask logit; a ray is kept when its
    sigmoid reaches the threshold. Dropped rays read zero range and zero
    intensity, matching what the reference tokenizer hands its consumers.
    """
    if network.ndim != 5 or network.shape[1] != 3:
        raise ValueError(
            f"Expected decoder output [B, 3, T, H, W], got {tuple(network.shape)}"
        )
    range_m = (network[:, 0:1].clamp(-1.0, 1.0) + 1.0) * 0.5 * (
        max_range_m - min_range_m
    ) + min_range_m
    intensity = (network[:, 1:2].clamp(-1.0, 1.0) + 1.0) * 0.5
    probability = torch.sigmoid(network[:, 2:3])
    if not apply_validity_mask:
        return torch.cat((range_m, intensity, probability), dim=1)
    valid = probability >= validity_threshold
    zeros = torch.zeros_like(range_m)
    return torch.cat(
        (
            torch.where(valid, range_m, zeros),
            torch.where(valid, intensity, zeros),
            valid.to(network.dtype),
        ),
        dim=1,
    )


def crop_lidar_width(clip: torch.Tensor, semantic_width: int) -> torch.Tensor:
    """Undo the circular azimuth padding: center-crop ``[..., W_model]`` to ``W_semantic``."""
    model_width = int(clip.shape[-1])
    if model_width < semantic_width or (model_width - semantic_width) % 2:
        raise ValueError(
            f"LiDAR model width {model_width} cannot be center-cropped to {semantic_width}."
        )
    offset = (model_width - semantic_width) // 2
    return clip[..., offset : offset + semantic_width] if offset else clip


class Cosmos3LidarDecoder(nn.Module):
    """Normalized LiDAR latents in, metric range clips out (FP32)."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = validate_lidar_config(config)
        network = self.config["network_config"]
        channels = int(self.config["latent_channels"])
        self.decoder = LidarTransformerDecoder(network)
        self.post_quant_conv = nn.Conv2d(channels, channels, kernel_size=1)
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

    @classmethod
    def from_pretrained(
        cls, model_path: str, config: Mapping[str, Any], device: torch.device | str
    ) -> Cosmos3LidarDecoder:
        from safetensors.torch import load_file

        folder = os.path.join(model_path, COSMOS3_LIDAR_COMPONENT)
        config_path = os.path.join(folder, "config.json")
        weights_path = os.path.join(folder, COSMOS3_LIDAR_WEIGHTS_FILE)
        if not os.path.isfile(config_path) or not os.path.isfile(weights_path):
            raise ValueError(
                f"Incomplete joint checkpoint: {COSMOS3_LIDAR_COMPONENT}/config.json and "
                f"{COSMOS3_LIDAR_WEIGHTS_FILE} are required to decode LiDAR."
            )
        with open(config_path) as handle:
            component = json.load(handle)
        _check_component_matches(component, config)
        model = cls(config).float()
        state = load_file(weights_path)
        wanted = {
            key: value
            for key, value in state.items()
            if key.startswith(("decoder.", "post_quant_conv."))
            or key in ("coords", "latent_mean", "latent_std")
        }
        missing, unexpected = model.load_state_dict(wanted, strict=False)
        if missing or unexpected:
            raise ValueError(
                "Cosmos3 LiDAR decoder weights do not match the exported component: "
                f"missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}."
            )
        logger.info(
            "Loaded Cosmos3 LiDAR decoder (%d tensors) from %s", len(wanted), folder
        )
        return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def decode_network(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized latents ``[B, C, T, H_z, W_z]`` to network-space ``[B, 3, T, H, W_model]``."""
        if latents.ndim != 5 or latents.shape[1] != self.latent_mean.shape[1]:
            raise ValueError(
                f"Expected LiDAR latents [B, {self.latent_mean.shape[1]}, T, H, W], "
                f"got {tuple(latents.shape)}"
            )
        with torch.autocast(device_type=self.coords.device.type, enabled=False):
            z = latents.to(device=self.coords.device, dtype=torch.float32)
            z = z * self.latent_std + self.latent_mean
            chunk = int(self.config["streaming_chunk_frames"])
            context = self.config["streaming_context_frames"]
            outputs = []
            cache = None
            for start in range(0, z.shape[2], chunk):
                piece = z[:, :, start : start + chunk]
                if cache is not None and context is not None:
                    keep = int(context) - piece.shape[2]
                    cache = (
                        {
                            key: (k[:, :, -keep:], v[:, :, -keep:])
                            for key, (k, v) in cache.items()
                        }
                        if keep > 0
                        else None
                    )
                batch, channels, frames, height, width = piece.shape
                flat = self.post_quant_conv(piece.permute(0, 2, 1, 3, 4).flatten(0, 1))
                piece = flat.view(batch, frames, channels, height, width).permute(
                    0, 2, 1, 3, 4
                )
                decoded, cache = self.decoder.forward_stream(piece, self.coords, cache)
                outputs.append(decoded)
            return torch.cat(outputs, dim=2)

    @torch.no_grad()
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized latents to the metric clip ``[B, 3, T, 128, 1800]``."""
        projection = self.projection
        metric = lidar_network_to_metric(
            self.decode_network(latents),
            min_range_m=float(projection["min_range_m"]),
            max_range_m=float(projection["max_range_m"]),
            validity_threshold=float(
                projection.get("validity_threshold", COSMOS3_LIDAR_VALIDITY_THRESHOLD)
            ),
            apply_validity_mask=bool(self.config.get("apply_validity_mask", True)),
        )
        return crop_lidar_width(metric, int(projection["semantic_width"]))
