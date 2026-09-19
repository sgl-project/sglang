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
so the state dict loads without remapping. Spatial neighborhood attention
requires the ``natten`` package.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    validate_lidar_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

COSMOS3_LIDAR_COMPONENT = "lidar_vae"
COSMOS3_LIDAR_WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"
# Range and intensity outside these predicates are encoded as the invalid fill.
COSMOS3_LIDAR_INVALID_VALUE = -1.0
COSMOS3_LIDAR_VALIDITY_THRESHOLD = 0.5
# Causal SDPA batches larger than this overflow the kernel's grid dimension.
_SDPA_MAX_BATCH = 65535


def _natten_na2d():
    try:
        from natten.functional import na2d
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Cosmos3 LiDAR encoding needs the natten package for neighborhood "
            "attention: pip install natten -f https://whl.natten.org"
        ) from exc
    return na2d


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
class LidarRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.scale).to(x.dtype)


class LidarAxialRoPE(nn.Module):
    """Rotary angles from the polar (elevation, azimuth) coordinates of each token."""

    def __init__(
        self, head_dim: int, num_heads: int, max_harmonics: tuple[int, int]
    ) -> None:
        super().__init__()
        quarter = head_dim // 4
        freqs_h = self._setup(num_heads * quarter, int(max_harmonics[0]))
        freqs_w = self._setup(num_heads * quarter, int(max_harmonics[1]))
        self.register_buffer("freqs_h", freqs_h.view(quarter, num_heads).T.contiguous())
        self.register_buffer("freqs_w", freqs_w.view(quarter, num_heads).T.contiguous())

    @staticmethod
    def _setup(dim: int, max_harmonics: int) -> torch.Tensor:
        return (
            torch.linspace(math.log(1), math.log(max(max_harmonics, 1)), dim)
            .exp()
            .round()
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """``[B, 2, H, W]`` coordinates to ``[B, H, W, heads, head_dim/2]`` angles."""
        coords = coords.permute(0, 2, 3, 1)
        radian_h = coords[..., None, 0:1] * self.freqs_h
        radian_w = coords[..., None, 1:2] * self.freqs_w
        return torch.cat((radian_h, radian_w), dim=-1)

    @staticmethod
    def rotate(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat(
            (x1 * theta.cos() - x2 * theta.sin(), x1 * theta.sin() + x2 * theta.cos()),
            dim=-1,
        )


def _scaled_unit_qk(
    q: torch.Tensor, k: torch.Tensor, scale: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    factor = scale.clamp(max=math.log(100)).exp().sqrt()
    q = (F.normalize(q, p=2, dim=-1, eps=eps) * factor).to(q.dtype)
    k = (F.normalize(k, p=2, dim=-1, eps=eps) * factor).to(k.dtype)
    return q, k


class LidarNeighborhoodAttention(nn.Module):
    """Spatial attention over a ``kernel_size`` neighborhood with a periodic azimuth axis."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int],
        max_harmonics: tuple[int, int],
        circular: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.kernel_size = tuple(int(k) for k in kernel_size)
        self.dilation = tuple(int(d) for d in dilation)
        self.circular = circular
        self.norm = LidarRMSNorm(dim)
        self.scale = nn.Parameter(torch.full((num_heads, 1), math.log(10.0)))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.rope = LidarAxialRoPE(self.head_dim, num_heads, max_harmonics)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = x.shape
        qkv = self.qkv_proj(self.norm(x)).view(
            batch, height, width, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=3)
        q, k = _scaled_unit_qk(q, k, self.scale, self.eps)
        theta = self.rope(coords)
        rotated = theta.shape[-1] * 2
        q = torch.cat(
            [self.rope.rotate(q[..., :rotated], theta).to(q.dtype), q[..., rotated:]],
            dim=-1,
        )
        k = torch.cat(
            [self.rope.rotate(k[..., :rotated], theta).to(k.dtype), k[..., rotated:]],
            dim=-1,
        )
        pad = self.kernel_size[1] // 2 if self.circular else 0
        if pad:
            q = F.pad(q, (0, 0, 0, 0, pad, pad), mode="circular")
            k = F.pad(k, (0, 0, 0, 0, pad, pad), mode="circular")
            v = F.pad(v, (0, 0, 0, 0, pad, pad), mode="circular")
        out = _natten_na2d()(
            query=q,
            key=k,
            value=v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            scale=1.0,
        )
        out = out.reshape(batch, height, out.shape[2], self.num_heads * self.head_dim)
        if pad:
            out = out[:, :, pad:-pad]
        return x + self.out_proj(out)


class LidarFeedForward(nn.Module):
    def __init__(self, dim: int, mid_dim: int) -> None:
        super().__init__()
        self.norm = LidarRMSNorm(dim)
        self.gegelu = nn.Linear(dim, mid_dim * 2, bias=False)
        self.linear = nn.Linear(mid_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, gate = self.gegelu(self.norm(x)).chunk(2, dim=-1)
        return x + self.linear(h * F.gelu(gate))


class LidarSpatialBlock(nn.Module):
    """Neighborhood attention + FFN over ``[B*T, H, W, C]``."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int],
        max_harmonics: tuple[int, int],
        mlp_ratio: float,
        circular: bool,
    ) -> None:
        super().__init__()
        self.residual_attn = LidarNeighborhoodAttention(
            dim, num_heads, kernel_size, dilation, max_harmonics, circular=circular
        )
        self.residual_ffn = LidarFeedForward(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.residual_ffn(self.residual_attn(x, coords))


def _chunked_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
) -> torch.Tensor:
    if q.shape[0] <= _SDPA_MAX_BATCH:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )
    return torch.cat(
        [
            F.scaled_dot_product_attention(
                q[i : i + _SDPA_MAX_BATCH],
                k[i : i + _SDPA_MAX_BATCH],
                v[i : i + _SDPA_MAX_BATCH],
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
            for i in range(0, q.shape[0], _SDPA_MAX_BATCH)
        ],
        dim=0,
    )


class LidarCausalTemporalAttention(nn.Module):
    """Causal attention along sweeps over ``[B*H*W, T, C]`` with a streaming K/V cache."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = LidarRMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, frames, _ = x.shape
        qkv = self.qkv_proj(self.norm(x)).view(
            rows, frames, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        return q, k, v

    def _finish(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return x + self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._qkv(x)
        return self._finish(x, _chunked_sdpa(q, k, v, None, True))

    def forward_stream(
        self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q, k_new, v_new = self._qkv(x)
        past = 0
        if kv_cache is not None:
            k_past, v_past = kv_cache
            past = k_past.shape[2]
            k = torch.cat([k_past, k_new], dim=2)
            v = torch.cat([v_past, v_new], dim=2)
        else:
            k, v = k_new, v_new
        new_frames = q.shape[2]
        q_index = past + torch.arange(new_frames, device=q.device)[:, None]
        k_index = torch.arange(past + new_frames, device=q.device)[None, :]
        out = _chunked_sdpa(q, k, v, k_index <= q_index, False)
        return self._finish(x, out), (k.detach(), v.detach())


class LidarTemporalBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.temporal_attn = LidarCausalTemporalAttention(dim, num_heads)
        self.ffn = LidarFeedForward(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.temporal_attn(x))

    def forward_stream(
        self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x, kv_cache = self.temporal_attn.forward_stream(x, kv_cache)
        return self.ffn(x), kv_cache


class LidarRope3D(nn.Module):
    """GPT-NeoX style rotary angles over a ``(T, H, W)`` grid, split per axis."""

    def __init__(
        self,
        head_dim: int,
        len_t: int,
        len_h: int,
        len_w: int,
        base_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        dim_h = (head_dim // 6) * 2
        dim_w = dim_h
        dim_t = head_dim - dim_h - dim_w
        if dim_t < 2:
            raise ValueError(f"head_dim={head_dim} is too small for 3D RoPE.")
        self.max_t, self.max_h, self.max_w = len_t, len_h, len_w
        for name, dim in (("freqs_t", dim_t), ("freqs_h", dim_h), ("freqs_w", dim_w)):
            idx = torch.arange(0, dim, 2, dtype=torch.float32) / dim
            self.register_buffer(name, 1.0 / (base_theta**idx), persistent=False)
        for name, length in (("seq_t", len_t), ("seq_h", len_h), ("seq_w", len_w)):
            self.register_buffer(
                name, torch.arange(length, dtype=torch.float32), persistent=False
            )

    def forward(self, frames: int, height: int, width: int) -> torch.Tensor:
        if frames > self.max_t or height > self.max_h or width > self.max_w:
            raise ValueError(
                f"LiDAR bottleneck grid ({frames},{height},{width}) exceeds the RoPE cache "
                f"({self.max_t},{self.max_h},{self.max_w})."
            )
        ang_t = torch.outer(self.seq_t[:frames], self.freqs_t).view(frames, 1, 1, -1)
        ang_h = torch.outer(self.seq_h[:height], self.freqs_h).view(1, height, 1, -1)
        ang_w = torch.outer(self.seq_w[:width], self.freqs_w).view(1, 1, width, -1)
        half = torch.cat(
            [
                ang_t.expand(frames, height, width, -1),
                ang_h.expand(frames, height, width, -1),
                ang_w.expand(frames, height, width, -1),
            ],
            dim=-1,
        )
        return torch.cat([half, half], dim=-1).reshape(frames * height * width, -1)


def _apply_rotary_3d(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate ``[..., S, heads, D]`` by ``[S, D]`` angles pairing ``i`` with ``i + D/2``."""
    angles = angles.to(x.dtype).unsqueeze(-2)
    cos, sin = angles.cos(), angles.sin()
    half = x.shape[-1] // 2
    x_lo, x_hi = x[..., :half], x[..., half:]
    return torch.cat(
        [
            x_lo * cos[..., :half] - x_hi * sin[..., :half],
            x_hi * cos[..., half:] + x_lo * sin[..., half:],
        ],
        dim=-1,
    )


class LidarJointAttention(nn.Module):
    """Causal joint attention over ``(T, H, W)`` bottleneck tokens with 3D RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_t: int,
        len_h: int,
        len_w: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.max_t = max_t
        self.norm = LidarRMSNorm(dim)
        self.scale = nn.Parameter(torch.full((num_heads, 1), math.log(10.0)))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.rope3d = LidarRope3D(self.head_dim, max_t, len_h, len_w)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames, height, width, _ = x.shape
        qkv = self.qkv_proj(self.norm(x)).view(
            batch, frames * height * width, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q, k = _scaled_unit_qk(q, k, self.scale, self.eps)
        return q, k, v

    def forward_stream(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, new_frames, height, width, _ = x.shape
        if new_frames > self.max_t:
            raise ValueError(
                f"LiDAR streaming chunk of {new_frames} sweeps exceeds bottleneck_3d_max_t={self.max_t}."
            )
        spatial = height * width
        q, k_new, v_new = self._qkv(x)
        # Cache unrotated K/V as [B, heads, T, S, D] so old sweeps can be dropped.
        k_new = k_new.view(
            batch, new_frames, spatial, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)
        v_new = v_new.view(
            batch, new_frames, spatial, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)
        if kv_cache is not None:
            k_all = torch.cat([kv_cache[0], k_new], dim=2)
            v_all = torch.cat([kv_cache[1], v_new], dim=2)
        else:
            k_all, v_all = k_new, v_new
        if k_all.shape[2] > self.max_t:
            k_all = k_all[:, :, -self.max_t :].contiguous()
            v_all = v_all[:, :, -self.max_t :].contiguous()
        total = k_all.shape[2]
        angles = self.rope3d(total, height, width).view(total, spatial, -1)
        q = _apply_rotary_3d(q, angles[-new_frames:].reshape(new_frames * spatial, -1))
        k = _apply_rotary_3d(
            k_all.permute(0, 2, 3, 1, 4).reshape(
                batch, total * spatial, self.num_heads, self.head_dim
            ),
            angles.reshape(total * spatial, -1),
        )
        v = v_all.permute(0, 2, 3, 1, 4).reshape(
            batch, total * spatial, self.num_heads, self.head_dim
        )
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
        mask = None
        if new_frames > 1:
            t_q = torch.arange(
                total - new_frames, total, device=q.device
            ).repeat_interleave(spatial)
            t_k = torch.arange(total, device=q.device).repeat_interleave(spatial)
            mask = t_k.unsqueeze(0) <= t_q.unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        out = out.permute(0, 2, 1, 3).reshape(batch, new_frames, height, width, -1)
        return x + self.out_proj(out), (k_all.detach(), v_all.detach())


class LidarBottleneckBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_t: int,
        len_h: int,
        len_w: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.attn = LidarJointAttention(dim, num_heads, max_t, len_h, len_w)
        self.ffn = LidarFeedForward(dim, int(dim * mlp_ratio))

    def forward_stream(
        self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x, kv_cache = self.attn.forward_stream(x, kv_cache)
        return self.ffn(x), kv_cache


class _SpaceToDepth(nn.Module):
    """``[B, 2H, 2W, C]`` to ``[B, H, W, 4C]`` in (row, column) patch order."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        x = x.view(batch, height // 2, 2, width // 2, 2, channels)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, height // 2, width // 2, 4 * channels
        )


class _ChannelsLast(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)


class _ChannelsFirst(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)


class LidarSpatialPE(nn.Module):
    def __init__(self, dim: int, resolution: tuple[int, int]) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, *resolution, dim))

    def forward(self) -> torch.Tensor:
        return self.embedding


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
        from safetensors.torch import load_file

        folder = os.path.join(model_path, COSMOS3_LIDAR_COMPONENT)
        config_path = os.path.join(folder, "config.json")
        weights_path = os.path.join(folder, COSMOS3_LIDAR_WEIGHTS_FILE)
        if not os.path.isfile(config_path) or not os.path.isfile(weights_path):
            raise ValueError(
                f"Incomplete joint checkpoint: {COSMOS3_LIDAR_COMPONENT}/config.json and "
                f"{COSMOS3_LIDAR_WEIGHTS_FILE} are required for LiDAR requests."
            )
        with open(config_path) as handle:
            component = json.load(handle)
        _check_component_matches(component, config)
        model = cls(config).float()
        state = load_file(weights_path)
        wanted = {
            key: value
            for key, value in state.items()
            if key.startswith(("encoder.", "quant_conv."))
            or key in ("coords", "latent_mean", "latent_std")
        }
        missing, unexpected = model.load_state_dict(wanted, strict=False)
        if missing or unexpected:
            raise ValueError(
                "Cosmos3 LiDAR encoder weights do not match the exported component: "
                f"missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}."
            )
        if (
            not torch.isfinite(model.latent_mean).all()
            or not torch.isfinite(model.latent_std).all()
            or bool((model.latent_std <= 0).any())
        ):
            raise ValueError(
                "Cosmos3 LiDAR latent statistics must be finite with positive std."
            )
        logger.info(
            "Loaded Cosmos3 LiDAR encoder (%d tensors) from %s", len(wanted), folder
        )
        return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)

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


def _check_component_matches(
    component: Mapping[str, Any], deployment: Mapping[str, Any]
) -> None:
    """The transformer's ``multiview.lidar`` block must agree with ``lidar_vae/config.json``."""
    for key, expected in deployment.items():
        actual = component.get(key)
        if key == "network_config":
            for sub_key, sub_expected in expected.items():
                if actual is None or actual.get(sub_key) != sub_expected:
                    raise ValueError(
                        f"LiDAR encoder metadata disagrees with the transformer contract on "
                        f"network_config.{sub_key}: {actual and actual.get(sub_key)!r} != {sub_expected!r}."
                    )
        elif actual != expected:
            raise ValueError(
                f"LiDAR encoder metadata disagrees with the transformer contract on {key}: "
                f"{actual!r} != {expected!r}."
            )
