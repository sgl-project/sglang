# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.platforms import current_platform


def clone_scheduler_runtime(scheduler: Any) -> Any:
    """Create an isolated scheduler runtime from a scheduler template or runtime."""
    return deepcopy(scheduler)


def get_or_create_request_scheduler(
    batch: Req, scheduler_template: Any, *, isolate: bool = False
) -> Any:
    """Return the scheduler runtime for this request.

    Diffusion serving currently executes one request at a time on the normal
    worker path, so reusing the stage-local scheduler preserves warmup caches
    and avoids unnecessary deepcopy overhead. Set ``isolate=True`` only when a
    request can run concurrently or outlive the stage-local scheduler state.
    """
    if batch.scheduler is None:
        batch.scheduler = (
            clone_scheduler_runtime(scheduler_template)
            if isolate
            else scheduler_template
        )
    return batch.scheduler


def pred_noise_to_pred_video(
    pred_noise: torch.Tensor,
    noise_input_latent: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: Any,
) -> torch.Tensor:
    """Convert predicted noise to clean latent."""
    if timestep.ndim == 2:
        timestep = timestep.flatten(0, 1)
        assert timestep.numel() == noise_input_latent.shape[0]
    elif timestep.ndim == 1:
        if timestep.shape[0] == 1:
            timestep = timestep.expand(noise_input_latent.shape[0])
        else:
            assert timestep.numel() == noise_input_latent.shape[0]
    else:
        raise ValueError(
            f"[pred_noise_to_pred_video] Invalid timestep shape: {timestep.shape}"
        )

    dtype = pred_noise.dtype
    device = pred_noise.device
    pred_noise = pred_noise.double().to(device)
    noise_input_latent = noise_input_latent.double().to(device)
    sigmas = scheduler.sigmas.double().to(device)
    high_dtype = (
        torch.float64 if current_platform.is_float64_supported() else torch.float32
    )
    timesteps = scheduler.timesteps.to(high_dtype).to(device)
    timestep_id = torch.argmin(
        (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
    )
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    pred_video = noise_input_latent - sigma_t * pred_noise
    return pred_video.to(dtype)
