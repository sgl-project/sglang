from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)


def rollout_scheduler_for(serving):
    """Some serving schedulers cannot be used for rollout; map them to one
    that can. Schedulers without a mapping pass through unchanged.
    """
    if isinstance(serving, FlowUniPCMultistepScheduler):
        return FlowMatchEulerDiscreteScheduler(shift=serving.config.shift)
    return serving
