# SPDX-License-Identifier: Apache-2.0
"""Shared spatial and temporal blocks for the Cosmos3 LiDAR VAE."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar.neighborhood_attention import (
    neighborhood_attention_2d,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


_SDPA_MAX_BATCH = 65535


_NATTEN_INSTALL_HINT = (
    "install the natten wheel built for this torch/CUDA pair: "
    "pip install natten==0.21.7 -f https://whl.natten.org"
)


_natten_backend_cache: dict[tuple[int, torch.dtype, int], str] = {}


def _natten_na2d():
    try:
        from natten.functional import na2d
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Cosmos3 LiDAR encoding needs the natten package for neighborhood "
            f"attention; {_NATTEN_INSTALL_HINT}"
        ) from exc
    return na2d


def resolve_natten_backend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> str:
    """natten kernel for these tensors; refuses the eager FlexAttention fallback.

    natten picks "flex-fna" when none of its compiled kernels can serve the
    call (no libnatten in the wheel, or fp32 on a GPU whose only kernel is the
    half-precision one). Without torch.compile that path materializes the full
    (tokens x tokens) score matrix: about 1 TB for one 20-sweep range-image
    chunk, so it must never run here.
    """
    key = (q.device.index or 0, q.dtype, q.shape[-1])
    backend = _natten_backend_cache.get(key)
    if backend is not None:
        return backend
    from natten.backends import choose_backend

    backend = choose_backend(q, k, v, torch_compile=False)
    if backend == "flex-fna":
        raise RuntimeError(
            "natten has no compiled neighborhood-attention kernel for "
            f"{q.dtype} on {q.device} (head_dim {q.shape[-1]}) and would fall "
            "back to its eager FlexAttention path, which materializes a dense "
            "score matrix and runs out of memory on the LiDAR range image; "
            f"{_NATTEN_INSTALL_HINT}"
        )
    logger.info("Cosmos3 LiDAR neighborhood attention: natten backend %s", backend)
    _natten_backend_cache[key] = backend
    return backend


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
        if envs.SGLANG_DIFFUSION_COSMOS3_LIDAR_USE_NATTEN:
            out = _natten_na2d()(
                query=q,
                key=k,
                value=v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                scale=1.0,
                backend=resolve_natten_backend(q, k, v),
            )
        else:
            out = neighborhood_attention_2d(
                q, k, v, kernel_size=self.kernel_size, dilation=self.dilation, scale=1.0
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


class _ChannelsLast(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)


class LidarSpatialPE(nn.Module):
    def __init__(self, dim: int, resolution: tuple[int, int]) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, *resolution, dim))

    def forward(self) -> torch.Tensor:
        return self.embedding
