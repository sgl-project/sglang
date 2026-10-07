# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any, List, Optional, Union

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.platforms import current_platform


def calculate_linear_shift(
    image_seq_len: int,
    *,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """return the affine dynamic shift used by native flow schedulers"""
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


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
