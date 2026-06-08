# SPDX-License-Identifier: Apache-2.0
"""3D rotary position embedding for OmniDreams (pure-torch NeoX port).

Faithful port of FlashDreams ``flashdreams.core.attention.rope`` +
``rope_kernel`` (the fused Triton kernel) into device-agnostic PyTorch so the
single-chunk and autoregressive forward paths run without the Triton dependency.

Key facts (verified against FlashDreams source for the distilled single-view
checkpoint):
- head_dim 128 splits T:H:W = 44:42:42 (``dim_h = dim_w = head_dim//6*2``).
- NeoX / non-interleaved rotation: pairs are ``(d, d + D/2)``.
- NTK extrapolation: theta = 10000 * ratio**(dim/(dim-2)); H/W ratio = 3.0, T = 1.0.
- ``shift_t(ar_idx)`` advances time positions by ``ar_idx * len_t`` (used for the
  KV-cache window in autoregressive rollout). Keys are rotated *before* being
  written to the cache (standard RoPE, not the cache-relative variant).
"""

from __future__ import annotations

import torch
from einops import repeat
from torch import Tensor

# OmniDreams uses non-interleaved (NeoX) rotation: the pair is (d, d + D/2).
ROPE_IS_NEOX_STYLE = True


def rope_dims(head_dim: int) -> tuple[int, int, int]:
    """Return the (T, H, W) split of a head dim for 3D RoPE.

    For head_dim=128 this yields (44, 42, 42): each spatial axis takes
    ``head_dim // 6 * 2`` and time takes the remainder.
    """
    dim_h = dim_w = head_dim // 6 * 2
    dim_t = head_dim - dim_h - dim_w
    return dim_t, dim_h, dim_w


def _compute_freqs(
    dim: int,
    extrapolation_ratio: float = 1.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Base RoPE frequencies for one axis with NTK extrapolation (shape [dim//2])."""
    dim_range = (
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim
    )
    ntk_factor = extrapolation_ratio ** (dim / (dim - 2))
    theta = 10000.0 * ntk_factor
    return 1.0 / (theta**dim_range)


class RotaryPositionEmbedding3D:
    """Standard 3D RoPE with unbounded autoregressive time positions.

    ``shift_t`` returns a full-width ``[L, 1, 1, head_dim]`` frequency tensor
    (memory layout (T, H, W)) suitable for :func:`apply_rope_freqs`.
    """

    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.head_dim = head_dim
        self.len_h = len_h
        self.len_w = len_w
        self.len_t = len_t
        self.device = device

        dim_w = dim_h = head_dim // 6 * 2
        dim_t = head_dim - (dim_h + dim_w)
        self.raw_freqs_h = _compute_freqs(dim_h, h_extrapolation_ratio, device)
        self.raw_freqs_w = _compute_freqs(dim_w, w_extrapolation_ratio, device)
        self.raw_freqs_t = _compute_freqs(dim_t, t_extrapolation_ratio, device)

        self.freqs_t, self.freqs_h, self.freqs_w = self._freq_components_for_len(len_t)

    def _freq_components_for_len(self, len_t: int) -> tuple[Tensor, Tensor, Tensor]:
        seq_t = torch.arange(len_t, dtype=torch.float32, device=self.device)
        seq_h = torch.arange(self.len_h, dtype=torch.float32, device=self.device)
        seq_w = torch.arange(self.len_w, dtype=torch.float32, device=self.device)
        freqs_t = repeat(
            torch.outer(seq_t, self.raw_freqs_t),
            "t d -> (t h w) 1 1 d",
            h=self.len_h,
            w=self.len_w,
        )
        freqs_h = repeat(
            torch.outer(seq_h, self.raw_freqs_h),
            "h d -> (t h w) 1 1 d",
            t=len_t,
            w=self.len_w,
        )
        freqs_w = repeat(
            torch.outer(seq_w, self.raw_freqs_w),
            "w d -> (t h w) 1 1 d",
            t=len_t,
            h=self.len_h,
        )
        return freqs_t, freqs_h, freqs_w

    def _cat_freqs(self, freqs_t: Tensor, freqs_h: Tensor, freqs_w: Tensor) -> Tensor:
        # Non-interleaved (NeoX): [t, h, w] repeated twice along the last dim.
        return torch.cat([freqs_t, freqs_h, freqs_w] * 2, dim=-1)

    def shift_t(self, autoregressive_index: int = 0) -> Tensor:
        """Frequencies for AR chunk ``autoregressive_index`` (offset = idx * len_t)."""
        offset = autoregressive_index * self.len_t
        freqs_t = self.freqs_t + offset * self.raw_freqs_t
        return self._cat_freqs(freqs_t, self.freqs_h, self.freqs_w)


def apply_rope_freqs(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply NeoX 3D RoPE to ``x``.

    Mirrors the FlashDreams fused kernel (non-interleaved branch):
        out[a] = x[a] * cos(f) - x[b] * sin(f)
        out[b] = x[b] * cos(f) + x[a] * sin(f)
    with ``(a, b) = (d, d + D/2)`` and ``f`` the first-half angles of ``freqs``.

    Args:
        x: ``[B, S, H, D]`` query or key.
        freqs: ``[S, 1, 1, D]`` full-width frequencies from ``shift_t`` (the
            first and second halves are identical by construction).
    Returns:
        Rotated tensor of shape ``[B, S, H, D]`` (cos/sin computed in fp32).
    """
    seq_len = freqs.shape[0]
    half = x.shape[-1] // 2
    f = freqs[..., :half].reshape(seq_len, half).view(1, seq_len, 1, half)
    cos = f.cos().to(x.dtype)
    sin = f.sin().to(x.dtype)
    a = x[..., :half]
    b = x[..., half:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)
