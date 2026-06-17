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
from typing import Optional

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


# ---------------------------------------------------------------------------
# The one context object (currently: just the parallel subsystem).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RuntimeContext:
    parallel: ParallelContext


def build_parallel_context(model_runner) -> ParallelContext:
    """Snapshot the resolved parallel topology — call ONCE, AFTER
    ``init_torch_distributed`` (groups + dp-attention all resolved by then).

    Sizes/ranks for grouped dimensions are read from the process GROUPS (the same
    source the legacy getters use, so the values are byte-equivalent); the attention
    dims come from the ``dp_attention`` getters; the engine ``dp_size`` (folded, no
    process group of its own) is read off the resolved ``model_runner``. Deferred
    imports avoid an import cycle (``parallel_state`` / ``dp_attention`` import early).
    """
    from sglang.srt.distributed.parallel_state import (
        get_attn_context_model_parallel_rank,
        get_attn_context_model_parallel_world_size,
        get_attn_tensor_model_parallel_rank,
        get_attn_tensor_model_parallel_world_size,
        get_moe_data_parallel_rank,
        get_moe_data_parallel_world_size,
        get_moe_expert_parallel_rank,
        get_moe_expert_parallel_world_size,
        get_moe_tensor_parallel_rank,
        get_moe_tensor_parallel_world_size,
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_world_size,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        get_world_rank,
        get_world_size,
    )
    from sglang.srt.layers.dp_attention import (
        get_attention_dp_rank,
        get_attention_dp_size,
    )

    mr = model_runner
    return ParallelContext(
        world_size=get_world_size(),
        world_rank=get_world_rank(),
        tp_size=get_tensor_model_parallel_world_size(),
        tp_rank=get_tensor_model_parallel_rank(),
        pp_size=get_pipeline_model_parallel_world_size(),
        pp_rank=get_pipeline_model_parallel_rank(),
        dp_size=mr.dp_size,
        dp_rank=mr.dp_rank,
        moe_ep_size=get_moe_expert_parallel_world_size(),
        moe_ep_rank=get_moe_expert_parallel_rank(),
        moe_dp_size=get_moe_data_parallel_world_size(),
        moe_dp_rank=get_moe_data_parallel_rank(),
        moe_tp_size=get_moe_tensor_parallel_world_size(),
        moe_tp_rank=get_moe_tensor_parallel_rank(),
        attn_tp_size=get_attn_tensor_model_parallel_world_size(),
        attn_tp_rank=get_attn_tensor_model_parallel_rank(),
        attn_cp_size=get_attn_context_model_parallel_world_size(),
        attn_cp_rank=get_attn_context_model_parallel_rank(),
        attn_dp_size=get_attention_dp_size(),
        attn_dp_rank=get_attention_dp_rank(),
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
