from typing import Any

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
    return get_or_create_request_scheduler(
        batch,
        scheduler,
        isolate=isolate and scheduler is serving_scheduler,
    )
