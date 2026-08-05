# SPDX-License-Identifier: Apache-2.0
"""Tensor operations shared by SelfLift progressive-resolution transitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_context as precision_autocast_context,
)
from sglang.multimodal_gen.runtime.utils.precision import (
    resolve_precision,
)

logger = init_logger(__name__)


def _append_dims(value: torch.Tensor, target_ndim: int) -> torch.Tensor:
    while value.ndim < target_ndim:
        value = value.unsqueeze(-1)
    return value


def flow_match_clean_sample(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Recover x0 from a flow-matching prediction: x0 = x_t - sigma * v."""
    sigma_tensor = torch.as_tensor(sigma, device=sample.device, dtype=sample.dtype)
    sigma_tensor = _append_dims(sigma_tensor, sample.ndim)
    return sample - sigma_tensor * model_output.to(sample.dtype)


def mix_selflift_latents(
    latent_upsample: torch.Tensor,
    pixel_upsample: torch.Tensor,
    *,
    mix_ratio: float,
    pixel_min: float,
    pixel_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend latent- and pixel-space lifts using SelfLift's sparse error mask."""
    if latent_upsample.shape != pixel_upsample.shape:
        raise ValueError(
            "SelfLift branches must have identical shapes, got "
            f"{tuple(latent_upsample.shape)} and {tuple(pixel_upsample.shape)}"
        )

    difference = (latent_upsample - pixel_upsample).abs().mean(dim=1, keepdim=True)
    threshold = torch.quantile(difference.flatten(1).float(), 1.0 - mix_ratio, dim=1)
    broadcast_shape = (-1,) + (1,) * (difference.ndim - 1)
    threshold = threshold.view(broadcast_shape).to(difference.dtype)
    selected = torch.where(
        difference >= threshold, difference, torch.zeros_like(difference)
    )

    selected_flat = selected.flatten(1)
    selected_min = (
        selected_flat.masked_fill(selected_flat == 0, float("inf"))
        .amin(dim=1)
        .view(broadcast_shape)
    )
    selected_max = selected.amax(dim=tuple(range(2, selected.ndim)), keepdim=True)
    normalized = torch.where(
        selected > 0,
        (selected - selected_min) / (selected_max - selected_min + 1e-6),
        selected,
    )
    pixel_weight = torch.where(
        normalized > 0,
        pixel_min + (pixel_max - pixel_min) * normalized,
        normalized,
    )
    mixed = (1.0 - pixel_weight) * latent_upsample + pixel_weight * pixel_upsample
    return mixed, pixel_weight


def vae_output_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return vae_output_tensor(output[0])
    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    raise TypeError(f"Unsupported VAE decode output: {type(output)!r}")


def vae_mode(output) -> torch.Tensor:
    latent_dist = getattr(output, "latent_dist", output)
    if isinstance(latent_dist, (tuple, list)) and latent_dist:
        latent_dist = latent_dist[0]
    mode = getattr(latent_dist, "mode", None)
    if callable(mode):
        return mode()
    if isinstance(latent_dist, torch.Tensor):
        return latent_dist
    mean = getattr(latent_dist, "mean", None)
    if isinstance(mean, torch.Tensor):
        return mean
    raise TypeError(f"Unsupported VAE encode output: {type(output)!r}")


def _align_scale_or_shift(
    value: torch.Tensor | float | int | None, reference: torch.Tensor
) -> torch.Tensor | float | int | None:
    if not isinstance(value, torch.Tensor):
        return value
    value = value.to(reference.device, reference.dtype)
    while value.ndim > reference.ndim:
        squeezed = False
        squeeze_order = list(range(2, value.ndim)) + [0, 1]
        for dim in squeeze_order:
            if value.shape[dim] == 1:
                value = value.squeeze(dim)
                squeezed = True
                break
        if not squeezed:
            break
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


def selflift_scale_shift(
    latents: torch.Tensor, pipeline_config: Any, vae
) -> tuple[torch.Tensor | float | int, torch.Tensor | float | int | None]:
    getter = getattr(pipeline_config, "get_decode_scale_and_shift", None)
    if getter is None:
        return 1.0, None
    scaling_factor, shift_factor = getter(latents.device, latents.dtype, vae)
    return (
        _align_scale_or_shift(scaling_factor, latents),
        _align_scale_or_shift(shift_factor, latents),
    )


def denormalize_vae_latent(
    latents: torch.Tensor, pipeline_config: Any, vae
) -> torch.Tensor:
    scaling_factor, shift_factor = selflift_scale_shift(latents, pipeline_config, vae)
    latents = latents / scaling_factor
    if shift_factor is not None:
        latents = latents + shift_factor
    return latents


def normalize_vae_latent(
    latents: torch.Tensor, pipeline_config: Any, vae
) -> torch.Tensor:
    scaling_factor, shift_factor = selflift_scale_shift(latents, pipeline_config, vae)
    if shift_factor is not None:
        latents = latents - shift_factor
    return latents * scaling_factor


def resize_spatial(
    tensor: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str,
    align_corners: bool | None = None,
) -> torch.Tensor:
    if tensor.ndim == 4:
        kwargs = {"size": size, "mode": mode}
        if align_corners is not None:
            kwargs["align_corners"] = align_corners
        return F.interpolate(tensor, **kwargs)
    if tensor.ndim == 5:
        B, C, T, H, W = tensor.shape
        flat = tensor.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        kwargs = {"size": size, "mode": mode}
        if align_corners is not None:
            kwargs["align_corners"] = align_corners
        resized = F.interpolate(flat, **kwargs)
        return resized.reshape(B, T, C, size[0], size[1]).permute(0, 2, 1, 3, 4)
    raise ValueError(
        f"SelfLift expects 4-D image or 5-D video latents, got {tensor.ndim}-D"
    )


def selflift_clean_upsample(
    *,
    clean_native: torch.Tensor,
    batch: Any,
    server_args: Any,
    vae,
    vae_dtype: torch.dtype,
    current_h_lat: int,
    current_w_lat: int,
    target_h_lat: int,
    target_w_lat: int,
    unpack_latent: Callable[[torch.Tensor, int, int], torch.Tensor],
    repack_latent: Callable[[torch.Tensor, int, int, Any, Any], torch.Tensor],
    spatial_to_vae_latent: Callable[[torch.Tensor, Any, Any, Any], torch.Tensor],
    vae_latent_to_spatial: Callable[[torch.Tensor, Any, Any, Any], torch.Tensor],
    latent_scale_factor: Callable[[Any], int],
    vae_latent_size: Callable[[int, int], tuple[int, int]],
    denormalize_vae_latent_fn: Callable[[torch.Tensor, Any, Any], torch.Tensor],
    normalize_vae_latent_fn: Callable[[torch.Tensor, Any, Any], torch.Tensor],
) -> torch.Tensor:
    clean_spatial = unpack_latent(clean_native, current_h_lat, current_w_lat)
    clean_vae_latent = spatial_to_vae_latent(clean_spatial, batch, server_args, vae)
    target_vae_size = vae_latent_size(target_h_lat, target_w_lat)

    latent_upsample = resize_spatial(clean_vae_latent, target_vae_size, mode="nearest")

    decode_latent = denormalize_vae_latent_fn(
        clean_vae_latent, server_args.pipeline_config, vae
    )
    decoded_output = vae.decode(decode_latent.to(vae_dtype))
    decoded = vae_output_tensor(decoded_output)

    target_pixel_size = (
        target_h_lat * latent_scale_factor(server_args),
        target_w_lat * latent_scale_factor(server_args),
    )
    decoded = resize_spatial(
        decoded, target_pixel_size, mode="bicubic", align_corners=False
    )

    encoded = vae_mode(vae.encode(decoded.to(vae_dtype))).to(latent_upsample.dtype)
    pixel_upsample = normalize_vae_latent_fn(encoded, server_args.pipeline_config, vae)
    if pixel_upsample.shape != latent_upsample.shape:
        raise RuntimeError(
            "VAE encode shape does not match SelfLift target: "
            f"{tuple(pixel_upsample.shape)} vs {tuple(latent_upsample.shape)}"
        )

    mixed, _ = mix_selflift_latents(
        latent_upsample,
        pixel_upsample,
        mix_ratio=float(getattr(batch, "selflift_mix_ratio", 0.4)),
        pixel_min=float(getattr(batch, "selflift_pixel_min", 0.7)),
        pixel_max=float(getattr(batch, "selflift_pixel_max", 1.0)),
    )
    mixed_spatial = vae_latent_to_spatial(
        mixed.to(clean_spatial.dtype), batch, server_args, vae
    )
    result = repack_latent(
        mixed_spatial, target_h_lat, target_w_lat, batch, server_args
    )
    return result


def _identity_latent_adapter(
    x: torch.Tensor, batch: Any, server_args: Any, vae
) -> torch.Tensor:
    del batch, server_args, vae
    return x


def _default_vae_latent_size(h_lat: int, w_lat: int) -> tuple[int, int]:
    return h_lat, w_lat


def apply_selflift_transition(
    *,
    stage: Any,
    ctx: Any,
    batch: Any,
    server_args: Any,
    sigma_t: float,
    seeds: Any,
    current_h_lat: int,
    current_w_lat: int,
    target_h_lat: int,
    target_w_lat: int,
) -> tuple[torch.Tensor, float | None]:
    prediction = ctx.extra.pop("progressive_last_prediction", None)
    if prediction is None:
        raise RuntimeError(
            "SelfLift reached a resolution transition before any denoising "
            "prediction. Increase the number of steps before the transition "
            "or reduce progressive_levels."
        )
    previous_sample, previous_model_output, previous_timestep = prediction
    if previous_model_output is None:
        raise RuntimeError(
            "SelfLift requires _run_denoising_step() to return the model prediction."
        )

    timestep_scale = float(
        getattr(getattr(ctx.scheduler, "config", None), "num_train_timesteps", 1000)
    )
    previous_sigma = previous_timestep.to(previous_sample.dtype) / timestep_scale
    clean_native = flow_match_clean_sample(
        previous_sample, previous_model_output, previous_sigma
    )
    if stage.vae is None:
        raise RuntimeError(f"{stage.__class__.__name__} SelfLift requires a VAE.")

    vae_dtype = resolve_precision(server_args, "vae", precision_attr="vae_precision")
    with stage.use_declared_component(
        component_name="vae",
        module=stage.vae,
        target_dtype=vae_dtype,
    ) as vae:
        assert vae is not None
        if not callable(getattr(vae, "encode", None)):
            raise RuntimeError(
                f"{stage.__class__.__name__} SelfLift requires a VAE encoder. "
                "Load an encoder-capable VAE for this request."
            )
        if getattr(server_args.pipeline_config, "vae_tiling", False):
            vae.enable_tiling()
        with precision_autocast_context(vae_dtype, server_args.disable_autocast):
            clean_high = selflift_clean_upsample(
                clean_native=clean_native,
                batch=batch,
                server_args=server_args,
                vae=vae,
                vae_dtype=vae_dtype,
                current_h_lat=current_h_lat,
                current_w_lat=current_w_lat,
                target_h_lat=target_h_lat,
                target_w_lat=target_w_lat,
                unpack_latent=stage._unpack_latent,
                repack_latent=stage._repack_latent,
                spatial_to_vae_latent=getattr(
                    stage,
                    "_selflift_spatial_to_vae_latent",
                    _identity_latent_adapter,
                ),
                vae_latent_to_spatial=getattr(
                    stage,
                    "_selflift_vae_latent_to_spatial",
                    _identity_latent_adapter,
                ),
                latent_scale_factor=stage._latent_scale_factor,
                vae_latent_size=getattr(
                    stage,
                    "_selflift_vae_latent_size",
                    _default_vae_latent_size,
                ),
                denormalize_vae_latent_fn=getattr(
                    stage,
                    "_selflift_denormalize_vae_latent",
                    denormalize_vae_latent,
                ),
                normalize_vae_latent_fn=getattr(
                    stage,
                    "_selflift_normalize_vae_latent",
                    normalize_vae_latent,
                ),
            )

    noise = randn_tensor(
        clean_high.shape,
        generator=stage._get_initial_noise_generator(batch, seeds, clean_high.device),
        device=clean_high.device,
        dtype=clean_high.dtype,
    )
    noise_shift = float(getattr(batch, "selflift_noise_shift", 0.97))
    shifted_sigma = sigma_t * noise_shift if noise_shift > 0 else sigma_t
    logger.info(
        "SelfLift transition: %dx%d → %dx%d latent, sigma=%.4f → %.4f",
        current_h_lat,
        current_w_lat,
        target_h_lat,
        target_w_lat,
        sigma_t,
        shifted_sigma,
    )
    return flow_match_renoise(clean_high, noise, shifted_sigma), shifted_sigma


def flow_match_renoise(
    clean_sample: torch.Tensor,
    noise: torch.Tensor,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Move a clean sample onto the flow-matching path at ``sigma``."""
    sigma_tensor = torch.as_tensor(
        sigma, device=clean_sample.device, dtype=clean_sample.dtype
    )
    sigma_tensor = _append_dims(sigma_tensor, clean_sample.ndim)
    return (1.0 - sigma_tensor) * clean_sample + sigma_tensor * noise
