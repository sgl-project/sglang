from typing import Any

import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


def rollout_scheduler_for(serving):
    """Some serving schedulers cannot be used for rollout; map them to one
    that can. Schedulers without a mapping pass through unchanged.
    """
    if isinstance(serving, FlowUniPCMultistepScheduler):
        return FlowMatchEulerDiscreteScheduler(shift=serving.config.shift)
    return serving


def get_or_create_rollout_request_scheduler(
    batch: Req,
    serving_scheduler: Any,
    *,
    isolate: bool = False,
) -> Any:
    """Return the scheduler runtime for a rollout request."""
    if batch.scheduler is not None:
        return batch.scheduler

    scheduler = rollout_scheduler_for(serving_scheduler)
    scheduler_is_shared = scheduler is serving_scheduler
    needs_clone = isolate and scheduler_is_shared
    return get_or_create_request_scheduler(
        batch,
        scheduler,
        isolate=needs_clone,
    )


def prepare_rollout_request_scheduler(
    batch: Req,
    serving_scheduler: Any,
    *,
    explicit_shift: float | None,
    num_inference_steps: int,
    device: torch.device,
) -> None:
    """Bind the per-request rollout scheduler and hand it a sigma grid.

    Without an explicit shift the rollout scheduler inherits the serving grid
    verbatim; an explicit shift selects a plain shifted grid instead.
    """
    scheduler = get_or_create_rollout_request_scheduler(batch, serving_scheduler)
    if scheduler is not serving_scheduler:
        if explicit_shift is not None:
            # Unwarped base; the mapped scheduler's sigmas are already serving-shift warped.
            shift = float(explicit_shift)
            num_train_timesteps = scheduler.config.num_train_timesteps
            sigmas = torch.linspace(1.0, 1.0 / num_train_timesteps, num_inference_steps)
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        else:
            sigmas = serving_scheduler.sigmas[:-1]
        # shift=1.0 so set_timesteps keeps the explicit sigmas verbatim.
        scheduler.set_shift(1.0)
        scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
    batch.timesteps = scheduler.timesteps
