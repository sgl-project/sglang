from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.host_state import (
    BatchPlan,
    plan_batch_from_forward_batch,
)
from sglang.srt.kv_cache_canary.pool_patch import PoolKind
from sglang.srt.kv_cache_canary.runner import CanaryRunner

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_GLOBAL_RUNNER_KEY = "_kv_cache_canary_runner"


def attach(
    *,
    pool: "KVCache",
    config: CanaryConfig,
    device: torch.device,
    pool_kind: PoolKind = PoolKind.FULL,
    verify_capacity: int,
    write_capacity: int,
    write_req_capacity: int,
) -> Optional[CanaryRunner]:
    """Attach canary to ``pool`` and create a runner.

    Must be called AFTER ``init_memory_pool`` and BEFORE ``init_device_graphs``
    so that the canary kernel is captured into the CUDA graph and the shadow
    tensors are baked into the graph's pointer table.

    The three capacities pre-size the fixed-address GPU launch buffers; the
    cuda graph records these specific addresses, so they must cover every
    forward shape (eager + replay).
    """
    if not config.enabled:
        return None
    if hasattr(pool, _GLOBAL_RUNNER_KEY):
        logger.warning("kv-canary: pool already has a runner attached; reusing it")
        return get_runner(pool)

    runner = CanaryRunner(
        config=config,
        pool=pool,
        device=device,
        pool_kind=pool_kind,
        verify_capacity=verify_capacity,
        write_capacity=write_capacity,
        write_req_capacity=write_req_capacity,
    )
    setattr(pool, _GLOBAL_RUNNER_KEY, runner)
    logger.info(
        "kv-canary: attached runner in mode=%s pool_kind=%s",
        config.mode.value,
        pool_kind.value,
    )
    return runner


def get_runner(pool: "KVCache") -> Optional[CanaryRunner]:
    return getattr(pool, _GLOBAL_RUNNER_KEY, None)


def run_head(
    *,
    runner: Optional[CanaryRunner],
    forward_batch: "ForwardBatch",
) -> Optional[BatchPlan]:
    if runner is None:
        return None
    plan = plan_batch_from_forward_batch(
        forward_batch=forward_batch, config=runner.config
    )
    if plan is None:
        return None
    runner.run_head(plan=plan)
    return plan


def run_tail(
    *,
    runner: Optional[CanaryRunner],
    plan: Optional[BatchPlan],
) -> None:
    if runner is None or plan is None:
        return
    runner.run_tail(plan=plan)
    runner.end_of_forward()


def launch_canary_for_capture(
    runner: Optional[CanaryRunner], *, kernel_kind: int
) -> None:
    """Capture-only kernel launch: record one kernel with skip-sentinel buffers."""
    if runner is None or not runner.config.enabled:
        return
    runner.launch_for_capture(kernel_kind=kernel_kind)


def prepare_replay(
    *,
    runner: Optional[CanaryRunner],
    forward_batch: "ForwardBatch",
) -> Optional[BatchPlan]:
    """Replay-side host hook: build the plan and refill the fixed buffers."""
    if runner is None or not runner.config.enabled:
        return None
    plan = plan_batch_from_forward_batch(
        forward_batch=forward_batch, config=runner.config
    )
    if plan is None:
        runner.reset_launch_buffers_to_skip_sentinel()
        return None
    runner.prepare_for_replay(plan=plan)
    return plan


def finalize_replay(
    *,
    runner: Optional[CanaryRunner],
    plan: Optional[BatchPlan],
) -> None:
    """Replay-side host hook: run end-of-forward AFTER replay."""
    if runner is None or not runner.config.enabled:
        return
    _ = plan  # plan is now stateless — kept for symmetry with prepare_replay.
    runner.end_of_forward()
