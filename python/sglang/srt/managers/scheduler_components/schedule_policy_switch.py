"""Runtime schedule policy switch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.managers.io_struct import (
    SchedulePolicySwitchReqInput,
    SchedulePolicySwitchReqOutput,
)
from sglang.srt.runtime_context import get_context, get_parallel

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def handle_schedule_policy_switch(
    scheduler: Scheduler, recv_req: SchedulePolicySwitchReqInput
) -> SchedulePolicySwitchReqOutput:
    """Replace the waiting-queue ranker before the next scheduling pass."""
    old_schedule_policy = scheduler.policy.policy.value
    new_schedule_policy = recv_req.new_schedule_policy.strip().lower()

    def _fail(msg: str) -> SchedulePolicySwitchReqOutput:
        logger.warning(
            "Schedule policy switch rejected (%s -> %s): %s",
            old_schedule_policy,
            new_schedule_policy,
            msg,
        )
        return SchedulePolicySwitchReqOutput(
            success=False,
            message=msg,
            old_schedule_policy=old_schedule_policy,
            new_schedule_policy=new_schedule_policy,
        )

    if scheduler._schedule_policy_switch_in_progress:
        return _fail("another schedule policy switch is already in progress")

    parallel = get_parallel()
    unsupported = [
        name
        for name, enabled in (
            ("pipeline parallelism", parallel.pp_size > 1),
            ("expert parallelism", parallel.ep_size > 1),
            ("DP attention", parallel.attn_dp_size > 1),
            (
                "attention context parallelism",
                parallel.attn_cp_size > 1 or parallel.enable_prefill_cp,
            ),
            ("decode context parallelism", parallel.dcp_size > 1),
            ("MoE data parallelism", parallel.moe_dp_size > 1),
            ("distributed weight data parallelism", parallel.dwdp_size > 1),
            ("sequence parallelism", parallel.enable_layernorm_sp),
        )
        if enabled
    ]
    if unsupported:
        return _fail(
            "runtime schedule policy switching currently supports only ordinary "
            "tensor and data parallelism; unsupported: " + ", ".join(unsupported)
        )

    if new_schedule_policy == old_schedule_policy:
        return SchedulePolicySwitchReqOutput(
            success=True,
            message="already in target policy",
            old_schedule_policy=old_schedule_policy,
            new_schedule_policy=new_schedule_policy,
        )

    scheduler._schedule_policy_switch_in_progress = True

    try:
        if scheduler.enable_priority_scheduling and new_schedule_policy not in (
            "fcfs",
            "lof",
        ):
            raise ValueError("priority scheduling requires schedule_policy=fcfs or lof")

        new_policy = scheduler._create_schedule_policy(
            new_schedule_policy, switching=True
        )
        # Cache settings can make the requested policy fall back to FCFS.
        new_schedule_policy = new_policy.policy.value
        get_context().override(
            "scheduler.schedule_policy_switch", schedule_policy=new_schedule_policy
        )

        scheduler.waiting_queue.set_on_leave(None)
        scheduler.policy = new_policy
        scheduler.schedule_policy = new_schedule_policy
        scheduler.maybe_wire_uniboost_gamma_ada()
        return SchedulePolicySwitchReqOutput(
            success=True,
            message="ok",
            old_schedule_policy=old_schedule_policy,
            new_schedule_policy=new_schedule_policy,
        )
    except Exception as e:
        return _fail(f"schedule policy switch failed: {e}")
    finally:
        scheduler._schedule_policy_switch_in_progress = False
