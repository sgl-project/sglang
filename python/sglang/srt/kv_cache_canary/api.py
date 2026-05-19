from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.host_state import (
    BatchPlan,
    plan_batch_from_forward_batch,
)
from sglang.srt.kv_cache_canary.pool_patch import (
    PoolKind,
    attach_canary_buffers,
    get_canary_buffer_groups,
)
from sglang.srt.kv_cache_canary.runner import CanaryRunner

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_GLOBAL_RUNNERS_KEY = "_kv_cache_canary_runners"


def attach(
    *,
    pool: "KVCache",
    config: CanaryConfig,
    device: torch.device,
    verify_capacity: int,
    write_capacity: int,
    write_req_capacity: int,
    req_to_token_pool: Optional["ReqToTokenPool"] = None,
) -> Optional[List[CanaryRunner]]:
    """Attach canaries to ``pool`` and create one runner per canary buffer group.

    For SWA-style pools this returns **two** runners (one ``FULL`` + one
    ``SWA``); for plain MHA / MLA pools it returns a single ``FULL``
    runner.

    Must be called AFTER ``init_memory_pool`` and BEFORE ``init_device_graphs``
    so that every canary kernel is captured into the CUDA graph and the canary buffer
    tensors are baked into the graph's pointer table.
    """
    if not config.enabled:
        return None
    if hasattr(pool, _GLOBAL_RUNNERS_KEY):
        logger.warning("kv-canary: pool already has runners attached; reusing them")
        return get_runners(pool)

    attach_canary_buffers(pool)
    buffer_groups = get_canary_buffer_groups(pool)
    if not buffer_groups:
        return None

    runners: List[CanaryRunner] = []
    for kind, group in buffer_groups.items():
        runner_config = _per_kind_config(config, kind=kind)
        runners.append(
            CanaryRunner(
                config=runner_config,
                buffer_group=group,
                device=device,
                verify_capacity=verify_capacity,
                write_capacity=write_capacity,
                write_req_capacity=write_req_capacity,
                req_to_token_pool=req_to_token_pool,
            )
        )
    setattr(pool, _GLOBAL_RUNNERS_KEY, runners)
    logger.info(
        "kv-canary: attached %d runner(s) in mode=%s kinds=%s",
        len(runners),
        config.mode.value,
        [r.pool_kind.value for r in runners],
    )
    return runners


def _per_kind_config(config: CanaryConfig, *, kind: PoolKind) -> CanaryConfig:
    """Specialise the shared CanaryConfig for one attention regime.

    The ``swa_window_size`` field gates verify-range clipping in the
    planner. Only the SWA canary clips; the FULL canary always covers
    the full prefix even when the parent pool is an SWA system, so we
    null out the window for it.
    """
    if kind is PoolKind.SWA:
        return config
    if config.swa_window_size is None:
        return config
    return dataclasses.replace(config, swa_window_size=None)


def get_runners(pool: "KVCache") -> Optional[List[CanaryRunner]]:
    """Return the list of runners attached to the pool, or ``None``."""
    return getattr(pool, _GLOBAL_RUNNERS_KEY, None)


def run_head(
    *,
    runners: Optional[List[CanaryRunner]],
    forward_batch: "ForwardBatch",
) -> Dict[PoolKind, BatchPlan]:
    """Build per-kind plans, launch each runner's head kernel.

    Returns a dict mapping each runner's pool kind to its plan; the same
    dict is passed verbatim back into :func:`run_tail` so the second pass
    skips re-computing identical inputs.
    """
    if not runners:
        return {}
    plans: Dict[PoolKind, BatchPlan] = {}
    for runner in runners:
        runner.set_last_forward_batch(forward_batch)
        plan = plan_batch_from_forward_batch(
            forward_batch=forward_batch, config=runner.config
        )
        if plan is None:
            continue
        runner.run_head(plan=plan)
        plans[runner.pool_kind] = plan
    return plans


def run_tail(
    *,
    runners: Optional[List[CanaryRunner]],
    plans: Dict[PoolKind, BatchPlan],
) -> None:
    """Launch each runner's tail kernel and run its end-of-forward bookkeeping."""
    if not runners:
        return
    for runner in runners:
        plan = plans.get(runner.pool_kind)
        if plan is None:
            continue
        runner.run_tail(plan=plan)
        runner.end_of_forward()


def launch_canary_for_capture(
    runners: Optional[List[CanaryRunner]], *, kernel_kind: int
) -> None:
    """Capture-only: record one kernel launch per attached runner."""
    if not runners:
        return
    for runner in runners:
        if not runner.config.enabled:
            continue
        runner.launch_for_capture(kernel_kind=kernel_kind)


def prepare_replay(
    *,
    runners: Optional[List[CanaryRunner]],
    forward_batch: "ForwardBatch",
) -> Dict[PoolKind, BatchPlan]:
    """Replay-side host hook: build the plan and refill the fixed buffers."""
    if not runners:
        return {}
    plans: Dict[PoolKind, BatchPlan] = {}
    for runner in runners:
        if not runner.config.enabled:
            continue
        plan = plan_batch_from_forward_batch(
            forward_batch=forward_batch, config=runner.config
        )
        if plan is None:
            runner.reset_launch_buffers_to_skip_sentinel()
            continue
        runner.prepare_for_replay(plan=plan)
        plans[runner.pool_kind] = plan
    return plans


def finalize_replay(
    *,
    runners: Optional[List[CanaryRunner]],
    plans: Dict[PoolKind, BatchPlan],
) -> None:
    """Replay-side host hook: run end-of-forward AFTER replay on every runner."""
    if not runners:
        return
    _ = plans  # plans are stateless — kept for symmetry with prepare_replay.
    for runner in runners:
        if not runner.config.enabled:
            continue
        runner.end_of_forward()
