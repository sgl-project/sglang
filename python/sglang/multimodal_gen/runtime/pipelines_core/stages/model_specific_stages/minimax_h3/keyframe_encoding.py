# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 keyframe (imgvid) condition encoding.

Condition anchor row recipe:

- ``video_vae.encode_images(PIL, use_fp16_latent=True)`` under a
  scoped seed-42 RNG fork — the DiagonalGaussian is SAMPLED
  (use_mean=False) with seed 42, so the seed is part of
  the contract, not a convenience
- normalize ``(z - latents_mean) / latents_std`` with the loader-injected
  ``MiniMaxH3VideoVAEArchConfig`` values
- patchify [1, 2, 2] into packed cond rows, fp32
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEArchConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42
MINIMAX_H3_KEYFRAME_PATCH_SIZE = (1, 2, 2)


@contextlib.contextmanager
def minimax_h3_scoped_encode_rng(seed: int, device: torch.device | None = None):
    """Seed torch RNGs for a deterministic sampled VAE encode without leaking state.

    The encode recipes seed the default torch generators right before a
    posterior-sampled VAE encode. Forking restores the process-global CPU and
    device generators after the encode while preserving the exact sampled
    result.
    """
    devices: list[torch.device] = []
    device_module = None
    device_type = None
    is_supported_backend = (
        current_platform.is_cuda()
        or current_platform.is_rocm()
        or current_platform.is_npu()
    )
    if (
        device is not None
        and is_supported_backend
        and device.type == current_platform.device_type
    ):
        device_module = torch.get_device_module(device)
        if device_module.is_available():
            devices = [device]
            device_type = current_platform.device_type
    fork_rng_context = (
        torch.random.fork_rng(devices=devices)
        if device_type is None
        else torch.random.fork_rng(devices=devices, device_type=device_type)
    )
    with fork_rng_context:
        torch.default_generator.manual_seed(int(seed))
        for forked_device in devices:
            assert device_module is not None
            with device_module.device(forked_device):
                device_module.manual_seed(int(seed))
        yield


@contextlib.contextmanager
def minimax_h3_scoped_encode_fp32(video_vae: Any):
    """Scope the video VAE to fp32 for one or more keyframe/reference encodes.

    encode_keyframe_cond_rows and encode_reference_video_rows each also guard
    their own cast (skipping it when already fp32), so nesting this around a
    caller that does more than one encode -- FL2VA's two keyframes, ref2va's
    image reference plus video reference -- turns their per-call casts into
    no-ops instead of toggling the whole VAE's dtype once per encode.
    """
    parameter = next(video_vae.parameters())
    prev_dtype = parameter.dtype
    if prev_dtype != torch.float32:
        video_vae.to(torch.float32)
    try:
        yield
    finally:
        if prev_dtype != torch.float32:
            video_vae.to(prev_dtype)


@functools.lru_cache(maxsize=None)
def _cached_latent_mean_std(
    mean_values: tuple[float, ...],
    std_values: tuple[float, ...],
    view_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU mean/std tensors for a fixed (values, shape) triple, built once.

    arch_config.latents_mean/std are static config values fixed for the
    life of a loaded VAE, so every call with the same values reconstructs
    an identical tensor; cache it instead of rebuilding on every encode.
    """
    mean = torch.tensor(mean_values).view(view_shape)
    std = torch.tensor(std_values).view(view_shape)
    return mean, std


@torch.inference_mode()
def minimax_h3_encode_keyframe_cond_rows(
    video_vae: Any,
    image: Any,
    arch_config: MiniMaxH3VideoVAEArchConfig,
) -> torch.Tensor:
    """Encode a target-canvas PIL image into packed imgvid cond rows.

    Returns [n_rows, 24 * patch_h * patch_w] fp32 on CPU.
    """
    seed = MINIMAX_H3_KEYFRAME_ENCODE_SEED
    # The encode recipe runs on fp32 weights. Normal H3 residency already keeps
    # the shared video VAE in fp32; retain the scoped cast for standalone use.
    parameter = next(video_vae.parameters())
    prev_dtype = parameter.dtype
    if prev_dtype != torch.float32:
        video_vae.to(torch.float32)
    try:
        with minimax_h3_scoped_encode_rng(seed, parameter.device):
            z = video_vae.encode_images(image, use_fp16_latent=True)[0]
    finally:
        if prev_dtype != torch.float32:
            video_vae.to(prev_dtype)
    z = z.cpu().float()
    if z.dim() == 4:
        z = z[None]
    latent_channels = arch_config.latent_channels
    if z.dim() != 5 or int(z.shape[1]) != latent_channels:
        raise ValueError(f"unexpected imgvid latent shape {list(z.shape)}")
    mean, std = _cached_latent_mean_std(
        tuple(arch_config.latents_mean),
        tuple(arch_config.latents_std),
        (1, latent_channels, 1, 1, 1),
    )
    z.sub_(mean).div_(std)
    rows = minimax_h3_patchify_video_latent(
        z, patch_size=list(MINIMAX_H3_KEYFRAME_PATCH_SIZE)
    )
    return rows.to(torch.float32)


__all__ = [
    "MINIMAX_H3_KEYFRAME_ENCODE_SEED",
    "MINIMAX_H3_KEYFRAME_PATCH_SIZE",
    "_cached_latent_mean_std",
    "minimax_h3_encode_keyframe_cond_rows",
    "minimax_h3_scoped_encode_rng",
    "minimax_h3_scoped_encode_fp32",
]
