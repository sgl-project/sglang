"""Sweep planning helpers.

Two callers feed the sweep path:

- :func:`collect_running_reqs_for_sweep` — grabs ``(req_pool_indices, seq_lens)`` from a forward batch (or
  the scheduler's running batch) so the runner can call
  :func:`sglang.srt.kv_cache_canary.plan_input.build_plan_input_running_sweep`.
- :func:`build_radix_orphan_input` — wraps the radix walker into the form the runner consumes for the
  separate radix-orphan sweep slot.

Both are host-side Python; sweep is a cold path so polars / list comprehension is acceptable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_cache_canary.plan_input import (
    PlanInput,
    build_plan_input_radix_sweep,
    build_plan_input_running_sweep,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunningSweepReqs:
    """Bundle of per-req tensors the running-sweep builder consumes.

    Both tensors live on the same device and have shape ``[bs]``. Padding rows (``req_pool_idx == 0``) are
    preserved verbatim — the plan kernel itself skips them.
    """

    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor


def collect_running_reqs_for_sweep(
    *, forward_batch: Optional["ForwardBatch"]
) -> Optional[RunningSweepReqs]:
    """Pull ``(req_pool_indices, seq_lens)`` from the last forward batch as the running-reqs source.

    Returns ``None`` when the forward batch is missing — the runner then skips the running-sweep
    contribution this cycle (radix sweep still runs).
    """
    if forward_batch is None:
        return None
    if forward_batch.req_pool_indices is None or forward_batch.seq_lens is None:
        return None
    return RunningSweepReqs(
        req_pool_indices=forward_batch.req_pool_indices,
        seq_lens=forward_batch.seq_lens,
    )


def build_running_sweep_input(
    *,
    req_to_token_pool: "ReqToTokenPool",
    running: RunningSweepReqs,
    extras_capacity: int,
) -> PlanInput:
    """Thin wrapper that forwards to the plan_input running-sweep builder.

    Kept separate from the per-forward builder because sweep capacity (and therefore the extras dummy
    tensors) is sized off the pool's max-slots upper bound, not the per-step max-token budget.
    """
    return build_plan_input_running_sweep(
        req_to_token_pool=req_to_token_pool,
        running_req_pool_indices=running.req_pool_indices,
        running_seq_lens=running.seq_lens,
        extras_capacity=extras_capacity,
    )


def build_radix_orphan_input(
    *,
    req_to_token_pool: "ReqToTokenPool",
    radix_cache: Optional["BasePrefixCache"],
    extras_capacity: int,
    swa_index_lut: Optional[torch.Tensor] = None,
) -> Optional[PlanInput]:
    """Build the radix-orphan-sweep :class:`PlanInput`, or ``None`` when no radix cache is wired.

    The walker itself lives in :mod:`sglang.srt.kv_cache_canary.plan_input` so the orphan extraction and
    materialization can be unit-tested without the sweep_planner indirection.
    """
    if radix_cache is None:
        return None
    return build_plan_input_radix_sweep(
        req_to_token_pool=req_to_token_pool,
        radix_cache=radix_cache,
        extras_capacity=extras_capacity,
        swa_index_lut=swa_index_lut,
    )
