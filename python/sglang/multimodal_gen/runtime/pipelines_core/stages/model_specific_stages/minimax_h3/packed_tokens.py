# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence

import torch


def _int_tuple(value: Sequence[int], name: str, length: int) -> tuple[int, ...]:
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}, got {list(value)!r}")
    out = tuple(int(item) for item in value)
    if any(item <= 0 for item in out):
        raise ValueError(f"{name} values must be positive, got {list(value)!r}")
    return out


def _rank(tensor: torch.Tensor, name: str, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape={list(tensor.shape)}")


def minimax_h3_patchify_video_latent(
    latent: torch.Tensor,
    *,
    patch_size: Sequence[int],
) -> torch.Tensor:
    """Pack SGLang video latent [B,C,T,H,W] into DiT token rows."""

    _rank(latent, "video latent", 5)
    pt, ph, pw = _int_tuple(patch_size, "patch_size", 3)
    batch, channel, full_t, full_h, full_w = (int(dim) for dim in latent.shape)
    if full_t % pt or full_h % ph or full_w % pw:
        raise ValueError(
            "video latent spatial/time dims must be divisible by patch_size: "
            f"shape={list(latent.shape)}, patch_size={[pt, ph, pw]}"
        )
    t, h, w = full_t // pt, full_h // ph, full_w // pw
    packed = latent.reshape(batch, channel, t, pt, h, ph, w, pw)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(batch * t * h * w, channel * pt * ph * pw).contiguous()


def minimax_h3_unpatchify_video_tokens(
    rows: torch.Tensor,
    *,
    latent_shape: Sequence[int],
    patch_size: Sequence[int],
) -> torch.Tensor:
    """Unpack DiT video token rows into SGLang latent [B,C,T,H,W]."""

    _rank(rows, "video token rows", 2)
    t, h, w, channel = _int_tuple(latent_shape, "latent_shape", 4)
    pt, ph, pw = _int_tuple(patch_size, "patch_size", 3)
    expected_dim = pt * ph * pw * channel
    if int(rows.shape[-1]) != expected_dim:
        raise ValueError(
            f"video token dim {int(rows.shape[-1])} != patch volume * channel "
            f"{expected_dim} for latent_shape={list(latent_shape)}, "
            f"patch_size={[pt, ph, pw]}"
        )
    rows_per_sample = t * h * w
    if int(rows.shape[0]) % rows_per_sample:
        raise ValueError(
            f"video rows {int(rows.shape[0])} must be divisible by t*h*w "
            f"{rows_per_sample} for latent_shape={list(latent_shape)}"
        )
    packed = rows.reshape(-1, t, h, w, channel, pt, ph, pw)
    latent = torch.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(-1, channel, t * pt, h * ph, w * pw).contiguous()


def minimax_h3_unpack_audio_tokens(
    rows: torch.Tensor,
    *,
    audio_t: int,
    audio_channel: int,
) -> torch.Tensor:
    """Unpack DiT audio token rows into SGLang audio VAE latent [C,latent_dim,T]."""

    _rank(rows, "audio token rows", 2)
    audio_t = int(audio_t)
    audio_channel = int(audio_channel)
    if audio_t <= 0 or audio_channel <= 0:
        raise ValueError(
            f"audio_t and audio_channel must be positive, got {audio_t=} "
            f"{audio_channel=}"
        )
    if int(rows.shape[0]) != audio_t:
        raise ValueError(f"audio rows {int(rows.shape[0])} != audio_t {audio_t}")
    if audio_t % audio_channel:
        raise ValueError(
            f"audio_t must be divisible by audio_channel, got {audio_t=} "
            f"{audio_channel=}"
        )
    native = rows.reshape(audio_channel, audio_t // audio_channel, int(rows.shape[-1]))
    return native.permute(0, 2, 1).contiguous()


__all__ = [
    "minimax_h3_patchify_video_latent",
    "minimax_h3_unpack_audio_tokens",
    "minimax_h3_unpatchify_video_tokens",
]
