from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)


def resolve_rollout_scheduler(serving):
    """Return the scheduler a rollout request denoises with (bound to batch.scheduler)."""
    if isinstance(serving, FlowUniPCMultistepScheduler):
        # UniPC cannot produce SDE log-probs; roll out with flow-match Euler
        # at the serving shift.
        return FlowMatchEulerDiscreteScheduler(shift=serving.config.shift)
    return serving
