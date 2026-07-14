# SPDX-License-Identifier: Apache-2.0
"""Serving-to-rollout scheduler equivalents.

The rollout SDE/log-prob math (SchedulerRLMixin) is defined on first-order
flow-match Euler dynamics, which not every serving scheduler provides. An
entry here asserts: for any pipeline serving the key scheduler class,
rolling out with the scheduler produced by the factory is validated.
Schedulers that are already RL-capable pass through unchanged; unmapped
ones are returned as-is and rejected later by RolloutDenoisingMixin
("does not support rollout").
"""

from __future__ import annotations

from typing import Any, Callable

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)


def _flow_unipc_to_euler(serving: FlowUniPCMultistepScheduler) -> Any:
    if serving.config.use_dynamic_shifting:
        raise ValueError(
            "The UniPC->Euler rollout mapping is validated for static shift only"
        )
    # config.shift is never None: FlowUniPCMultistepScheduler asserts on it.
    return FlowMatchEulerDiscreteScheduler(shift=serving.config.shift)


_ROLLOUT_EQUIVALENTS: dict[type, Callable[[Any], Any]] = {
    FlowUniPCMultistepScheduler: _flow_unipc_to_euler,
}


def resolve_rollout_scheduler(serving: Any) -> Any:
    """Return the scheduler a rollout request should denoise with.

    The returned scheduler is owned by the request (bound to
    ``batch.scheduler``); the engine keeps only its serving scheduler.
    """
    if isinstance(serving, SchedulerRLMixin):
        return serving
    factory = _ROLLOUT_EQUIVALENTS.get(type(serving))
    return factory(serving) if factory is not None else serving
