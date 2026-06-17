"""One structured container for process-static runtime state.

This is the seed of the unified "Global Context Object" — a single
``RuntimeContext``, accessed via ``get_context()``, that replaces process-level
state scattered across many modules. It currently owns ONE subsystem and grows by
subsystem over time:

  ctx.parallel : the resolved parallel topology — every dimension's size + rank.

The parallel topology is resolved ONCE (after ``init_torch_distributed``) and read
through ``get_parallel()``. The scattered ``parallel_state`` size/rank getters become
thin shims over it (names/signatures unchanged), so call-sites do not change.

Concurrency: ``_context`` is a plain module-level global, safe because a worker
process resolves and runs synchronously on one thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Parallel topology: every dimension's resolved size + rank, in one object.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParallelContext:
    """The resolved parallel topology of this process. Each dimension's size + rank,
    snapshotted once after distributed init from the canonical sources (the process
    groups + dp-attention). Replaces the ~16 ``parallel_state`` getter functions that
    each independently reached into a group; those become thin readers over this
    object. Fields default so the container can be built empty (tests).

    Sizes/ranks here are byte-equivalent to what the legacy getters returned
    (``get_tp_group().world_size`` / ``.rank_in_group``); the snapshot is taken from
    the groups, which are static after init.
    """

    # --- world ---
    world_size: int = 1
    world_rank: int = 0
    # --- tensor parallel ---
    tp_size: int = 1
    tp_rank: int = 0
    # --- pipeline parallel ---
    pp_size: int = 1
    pp_rank: int = 0
    # --- data parallel (engine; folded by enable_dp_attention) ---
    dp_size: int = 1
    dp_rank: Optional[int] = None
    # --- MoE ---
    moe_ep_size: int = 1
    moe_ep_rank: int = 0
    moe_dp_size: int = 1
    moe_dp_rank: int = 0
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    # --- attention parallel (resolved by initialize_dp_attention) ---
    attn_tp_size: int = 1
    attn_tp_rank: int = 0
    attn_cp_size: int = 1
    attn_cp_rank: int = 0
    attn_dp_size: int = 1
    attn_dp_rank: int = 0
    local_attn_dp_size: int = 1
    local_attn_dp_rank: int = 0
    # --- process-group handles (the SAME GroupCoordinator objects the legacy
    # globals reference, so identity — incl. _MOE_DP/_ATTN_CP aliasing _TP — is
    # preserved). tp_group is intentionally NOT held: get_tp_group() is runtime-
    # dynamic under PD-multiplexing (returns _PDMUX_PREFILL_TP_GROUP), so it can't be
    # a static snapshot. Group `destroy` stays in parallel_state; reset_context drops
    # these references. ---
    world_group: Optional[Any] = None
    pp_group: Optional[Any] = None
    moe_ep_group: Optional[Any] = None
    moe_dp_group: Optional[Any] = None
    moe_tp_group: Optional[Any] = None
    attn_tp_group: Optional[Any] = None
    attn_cp_group: Optional[Any] = None


# ---------------------------------------------------------------------------
# The one context object (currently: just the parallel subsystem).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RuntimeContext:
    parallel: ParallelContext


def build_parallel_context(model_runner) -> ParallelContext:
    """Snapshot the resolved parallel topology — call ONCE, AFTER
    ``init_torch_distributed`` (groups + dp-attention all resolved by then).

    Sizes/ranks for grouped dimensions are read from the process GROUPS directly
    (``get_X_group().world_size`` / ``.rank_in_group`` — exactly what the legacy
    getters returned, so the values are byte-equivalent); reading the groups rather
    than the (now context-backed) size getters keeps the snapshot robust under
    multi-runner re-publish. The attention ``dp`` dim comes from the ``dp_attention``
    getters; the engine ``dp_size`` (folded, no process group of its own) is read off
    the resolved ``model_runner``. Deferred imports avoid an import cycle
    (``parallel_state`` / ``dp_attention`` import early)."""
    from sglang.srt.distributed.parallel_state import (
        get_attn_cp_group,
        get_attn_tp_group,
        get_moe_dp_group,
        get_moe_ep_group,
        get_moe_tp_group,
        get_pp_group,
        get_tp_group,
        get_world_group,
    )
    from sglang.srt.layers.dp_attention import (
        get_attention_dp_rank,
        get_attention_dp_size,
        get_local_attention_dp_rank,
        get_local_attention_dp_size,
    )

    mr = model_runner
    world, tp, pp = get_world_group(), get_tp_group(), get_pp_group()
    moe_ep, moe_dp, moe_tp = get_moe_ep_group(), get_moe_dp_group(), get_moe_tp_group()
    attn_tp, attn_cp = get_attn_tp_group(), get_attn_cp_group()
    return ParallelContext(
        world_size=world.world_size,
        world_rank=world.rank_in_group,
        tp_size=tp.world_size,
        tp_rank=tp.rank_in_group,
        pp_size=pp.world_size,
        pp_rank=pp.rank_in_group,
        dp_size=mr.dp_size,
        dp_rank=mr.dp_rank,
        moe_ep_size=moe_ep.world_size,
        moe_ep_rank=moe_ep.rank_in_group,
        moe_dp_size=moe_dp.world_size,
        moe_dp_rank=moe_dp.rank_in_group,
        moe_tp_size=moe_tp.world_size,
        moe_tp_rank=moe_tp.rank_in_group,
        attn_tp_size=attn_tp.world_size,
        attn_tp_rank=attn_tp.rank_in_group,
        attn_cp_size=attn_cp.world_size,
        attn_cp_rank=attn_cp.rank_in_group,
        attn_dp_size=get_attention_dp_size(),
        attn_dp_rank=get_attention_dp_rank(),
        local_attn_dp_size=get_local_attention_dp_size(),
        local_attn_dp_rank=get_local_attention_dp_rank(),
        # group handles (same objects as the locals above; tp_group excluded — PDMUX)
        world_group=world,
        pp_group=pp,
        moe_ep_group=moe_ep,
        moe_dp_group=moe_dp,
        moe_tp_group=moe_tp,
        attn_tp_group=attn_tp,
        attn_cp_group=attn_cp,
    )


# ---------------------------------------------------------------------------
# Module-global lifecycle (single publish / single teardown).
# ---------------------------------------------------------------------------

_context: Optional[RuntimeContext] = None


def set_context(ctx: RuntimeContext) -> None:
    """Low-level publish primitive (overwrite the module global). A second
    ModelRunner re-publishes a fresh context, so multi-runner is safe."""
    global _context
    _context = ctx


def has_context() -> bool:
    return _context is not None


def get_context() -> RuntimeContext:
    if _context is None:
        raise ValueError(
            "RuntimeContext not initialized — build it (build_parallel_context) "
            "and publish via set_context() first"
        )
    return _context


def reset_context() -> None:
    """Single teardown point (legacy globals have no such primitive — a
    multi-engine / test-isolation hazard)."""
    global _context
    _context = None


def get_parallel() -> ParallelContext:
    """The parallel-topology namespace — read leaves directly:
    ``get_parallel().tp_size`` / ``get_parallel().attn_dp_rank``."""
    return get_context().parallel
