from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.jit_kernel import kv_cache_canary_plan_ref as _canary_plan_ref
from sglang.jit_kernel.kv_cache_canary_plan_ref import (
    BatchPlan,
    plan_batch_from_forward_batch,
)
from sglang.srt.kv_cache_canary.config import CanaryConfig
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

_USE_TRITON_PLAN_ENV = "SGLANG_KV_CANARY_PLAN_USE_TRITON"
_PSEUDO_MODE_PLAN_PATCHED_ATTR = "_pseudo_mode_plan_patched"


def _triton_plan_enabled() -> bool:
    """Pick the Triton plan path unless explicitly disabled or pseudo-mode patched.

    Triton is the default — it avoids the 4 D2H + per-req Python loop in
    the reference impl. Two opt-outs:

    - ``SGLANG_KV_CANARY_PLAN_USE_TRITON=0`` env var (debug escape hatch).
    - pseudo-mode install has monkey-patched
      ``kv_cache_canary_plan_ref.plan_batch_from_forward_batch`` (it wraps
      the ref to inject ``expected_write_*`` and the Triton kernel does
      not yet implement that oracle). The patch sets a sentinel attr
      on the ref module that we detect here.
    """
    if os.environ.get(_USE_TRITON_PLAN_ENV, "1") == "0":
        return False
    if getattr(_canary_plan_ref, _PSEUDO_MODE_PLAN_PATCHED_ATTR, False):
        return False
    return True


def attach(
    *,
    pool: "KVCache",
    config: CanaryConfig,
    device: torch.device,
    verify_capacity: int,
    write_capacity: int,
    write_req_capacity: int,
    req_to_token_pool: Optional["ReqToTokenPool"] = None,
    tp_rank: int = 0,
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
                tp_rank=tp_rank,
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
) -> Dict[PoolKind, Optional[BatchPlan]]:
    """Build per-kind plans, launch each runner's head kernel.

    Returns a dict mapping each runner's pool kind to its host-side
    :class:`BatchPlan` (ref path) or ``None`` (Triton path — the plan
    lives only on device in ``runner._launch``). The same dict is
    passed verbatim back into :func:`run_tail`; presence of a key
    signals "this runner saw a valid forward this step", value is the
    plan only when the ref path needs it for the tail refill.
    """
    if not runners:
        return {}
    use_triton = _triton_plan_enabled()
    plans: Dict[PoolKind, Optional[BatchPlan]] = {}
    for runner in runners:
        runner.set_last_forward_batch(forward_batch)
        if use_triton:
            ok = runner.fill_launch_from_forward_batch_triton(
                forward_batch=forward_batch
            )
            if not ok:
                continue
            runner.launch_head_only()
            plans[runner.pool_kind] = None
        else:
            plan = plan_batch_from_forward_batch(
                forward_batch=forward_batch,
                config=runner.config,
                swa_index_lut=runner.buffer_group.swa_index_lut,
            )
            if plan is None:
                continue
            runner.run_head(plan=plan)
            plans[runner.pool_kind] = plan
    return plans


def run_tail(
    *,
    runners: Optional[List[CanaryRunner]],
    plans: Dict[PoolKind, Optional[BatchPlan]],
) -> None:
    """Launch each runner's tail kernel and run its end-of-forward bookkeeping."""
    if not runners:
        return
    for runner in runners:
        if runner.pool_kind not in plans:
            continue
        plan = plans[runner.pool_kind]
        if plan is None:
            runner.launch_tail_only()
        else:
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
) -> Dict[PoolKind, Optional[BatchPlan]]:
    """Replay-side host hook: build the plan and refill the fixed buffers."""
    if not runners:
        return {}
    use_triton = _triton_plan_enabled()
    plans: Dict[PoolKind, Optional[BatchPlan]] = {}
    for runner in runners:
        if not runner.config.enabled:
            continue
        if use_triton:
            ok = runner.fill_launch_from_forward_batch_triton(
                forward_batch=forward_batch
            )
            if not ok:
                runner.reset_launch_buffers_to_skip_sentinel()
                continue
            plans[runner.pool_kind] = None
        else:
            plan = plan_batch_from_forward_batch(
                forward_batch=forward_batch,
                config=runner.config,
                swa_index_lut=runner.buffer_group.swa_index_lut,
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
    plans: Dict[PoolKind, Optional[BatchPlan]],
) -> None:
    """Replay-side host hook: run end-of-forward AFTER replay on every runner."""
    if not runners:
        return
    _ = plans  # plans are stateless — kept for symmetry with prepare_replay.
    for runner in runners:
        if not runner.config.enabled:
            continue
        runner.end_of_forward()
