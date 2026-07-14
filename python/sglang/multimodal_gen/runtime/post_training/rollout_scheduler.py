from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)


def rollout_scheduler_for(serving):
    """Map a pipeline's serving scheduler to the one rollout requests denoise with.

    Timestep-preparation stages call this under ``batch.rollout`` before
    binding ``batch.scheduler``; the returned scheduler is owned by the
    request. The rollout SDE/log-prob math (SchedulerRLMixin) runs on
    flow-match Euler dynamics, so serving schedulers that cannot provide it
    map to an Euler reproducing their schedule — one isinstance branch per
    validated scheduler family. Schedulers without a branch pass through
    unchanged and are rejected by RolloutDenoisingMixin ("does not support
    rollout") when a rollout request reaches them.
    """
    if isinstance(serving, FlowUniPCMultistepScheduler):
        # UniPC cannot produce SDE log-probs; roll out with flow-match Euler
        # at the serving shift.
        return FlowMatchEulerDiscreteScheduler(shift=serving.config.shift)
    return serving
