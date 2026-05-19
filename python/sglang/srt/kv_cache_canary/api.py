"""Facade for the canary API used by other srt modules.

External surface:

- :func:`attach` — install canary buffers + create a :class:`CanaryRunner` per attached
  :class:`CanaryBufferGroup`.
- :func:`get_runners` — fetch the runners attached to a pool.
- :func:`attach_radix_cache_to_pool` — late-binding helper invoked once the scheduler has built its
  ``tree_cache`` (the canary's :func:`attach` runs at ``ModelRunner`` init, before the radix cache exists).
- :func:`run_head` / :func:`run_tail` — per-forward driver delegating to each runner.
- :func:`launch_canary_for_capture` — capture-only no-op recording.
- :func:`prepare_replay` / :func:`finalize_replay` — replay-side hooks for cuda-graph runners.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.kv_cache_canary.buffer_group import PoolKind
from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.pool_patch import (
    attach_canary_buffers,
    get_canary_buffer_groups,
)
from sglang.srt.kv_cache_canary.runner import CanaryRunner

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_GLOBAL_RUNNERS_KEY = "_kv_cache_canary_runners"


def attach(
    *,
    pool: "KVCache",
    config: CanaryConfig,
    device: torch.device,
    per_forward_verify_capacity: int,
    per_forward_write_req_capacity: int,
    running_sweep_verify_capacity: int,
    radix_sweep_verify_capacity: int,
    radix_sweep_extras_capacity: int,
    per_forward_extras_capacity: int = 1,
    running_sweep_extras_capacity: int = 1,
    pseudo_token_capacity: int = 1,
    req_to_token_pool: Optional["ReqToTokenPool"] = None,
    radix_cache: Optional["BasePrefixCache"] = None,
    tp_rank: int = 0,
) -> Optional[List[CanaryRunner]]:
    """Attach canaries to ``pool`` and create one runner per attached canary buffer group.

    For SWA-style pools this returns **two** runners (one ``FULL`` + one ``SWA``); for plain MHA / MLA
    pools it returns a single ``FULL`` runner.

    Must be called AFTER ``init_memory_pool`` and BEFORE ``init_device_graphs`` so every canary kernel is
    captured into the CUDA graph and the canary buffer tensors are baked into the graph's pointer table.
    """
    if not config.enabled:
        return None
    if hasattr(pool, _GLOBAL_RUNNERS_KEY):
        logger.warning("kv-canary: pool already has runners attached; reusing them")
        return get_runners(pool)

    attach_canary_buffers(pool, real_kv_read_bytes=config.real_kv_read_bytes)
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
                per_forward_verify_capacity=per_forward_verify_capacity,
                per_forward_write_req_capacity=per_forward_write_req_capacity,
                running_sweep_verify_capacity=running_sweep_verify_capacity,
                radix_sweep_verify_capacity=radix_sweep_verify_capacity,
                radix_sweep_extras_capacity=radix_sweep_extras_capacity,
                per_forward_extras_capacity=per_forward_extras_capacity,
                running_sweep_extras_capacity=running_sweep_extras_capacity,
                pseudo_token_capacity=pseudo_token_capacity,
                req_to_token_pool=req_to_token_pool,
                radix_cache=radix_cache,
                tp_rank=tp_rank,
            )
        )
    setattr(pool, _GLOBAL_RUNNERS_KEY, runners)
    logger.info(
        "kv-canary: attached %d runner(s) in mode=%s kinds=%s",
        len(runners),
        config.mode.value,
        [r.pool_kind.name for r in runners],
    )
    return runners


def _per_kind_config(config: CanaryConfig, *, kind: PoolKind) -> CanaryConfig:
    """Specialise the shared :class:`CanaryConfig` for one attention regime.

    The ``swa_window_size`` field gates verify-range clipping in the plan kernel. Only the SWA canary
    clips; the FULL canary always covers the full prefix even when the parent pool is an SWA system, so we
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


def attach_radix_cache_to_pool(pool: "KVCache", radix_cache: "BasePrefixCache") -> None:
    """Bind ``radix_cache`` to every canary runner attached to ``pool``.

    No-op when canary is disabled or runners are not attached — keeps the scheduler call site cheap to
    invoke unconditionally. Required for radix-orphan sweep coverage (SOT §6.2): the radix cache is built
    by the scheduler after :func:`attach` already ran at ``ModelRunner`` init, so the runner needs a
    late-binding hook.
    """
    runners = get_runners(pool)
    if not runners:
        return
    for runner in runners:
        runner.attach_radix_cache(radix_cache)
    logger.info(
        "kv-canary: bound radix cache %s to %d runner(s)",
        type(radix_cache).__name__,
        len(runners),
    )


def run_head(
    *,
    runners: Optional[List[CanaryRunner]],
    forward_batch: "ForwardBatch",
) -> None:
    """Drive each runner's per-forward HEAD launch."""
    if not runners:
        return
    for runner in runners:
        runner.set_last_forward_batch(forward_batch)
        runner.run_head(forward_batch=forward_batch)


def run_tail(
    *,
    runners: Optional[List[CanaryRunner]],
    forward_batch: "ForwardBatch",
) -> None:
    """Drive each runner's per-forward TAIL launch + end-of-forward bookkeeping."""
    if not runners:
        return
    for runner in runners:
        runner.run_tail(forward_batch=forward_batch)
        runner.end_of_forward()


def launch_canary_for_capture(
    runners: Optional[List[CanaryRunner]], *, kernel_kind: str
) -> None:
    """Capture-only no-op recording hook.

    The kernel pair is intentionally NOT baked into cuda-graph capture (see :mod:`install`); this stays in
    place as the public symbol for any callers that still drive it. ``kernel_kind`` is accepted for
    backward-compat and ignored.
    """
    if not runners:
        return
    _ = kernel_kind


def prepare_replay(
    *,
    runners: Optional[List[CanaryRunner]],
    forward_batch: "ForwardBatch",
) -> None:
    """Replay-side hook: drive head launch + plan refill before the captured replay runs."""
    if not runners:
        return
    for runner in runners:
        if not runner.config.enabled:
            continue
        runner.set_last_forward_batch(forward_batch)
        runner.run_head(forward_batch=forward_batch)


def finalize_replay(
    *,
    runners: Optional[List[CanaryRunner]],
    forward_batch: "ForwardBatch",
) -> None:
    """Replay-side hook: drive tail launch + end-of-forward after the captured replay returns."""
    if not runners:
        return
    for runner in runners:
        if not runner.config.enabled:
            continue
        runner.run_tail(forward_batch=forward_batch)
        runner.end_of_forward()
