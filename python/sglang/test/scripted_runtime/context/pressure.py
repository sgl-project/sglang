"""Free functions for deterministic KV / pool pressure injection.

These let a script drive OOM / starvation branches without engineering a
real workload. Each takes the facade ``ctx`` first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def exhaust_kv(ctx: "ScriptedContext", *, leave_pages: int) -> None:
    """Consume KV pages until only ``leave_pages`` remain free.

    Deterministic KV pressure injection: occupies pages with dummy
    allocations so the next admit / extend hits the OOM branch
    without needing to engineer a real workload.

    Consumed by: test_chunked_retract_at_chunk_first_mid_last (kv_pressure),
                 test_kv_oom_triggers_retract (kv_pressure).
    """
    raise NotImplementedError(
        "scripted_runtime: exhaust_kv is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def exhaust_row_pool(ctx: "ScriptedContext", *, leave_rows: int) -> None:
    """Consume row-pool entries until only ``leave_rows`` remain free.

    Row-pool analogue of :meth:`exhaust_kv` — drives the row-pool
    starvation branch independently of token-level KV pressure.

    Consumed by: test_row_pool_starvation_blocks_admit (kv_pressure).
    """
    raise NotImplementedError(
        "scripted_runtime: exhaust_row_pool is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def exhaust_lock_refs(ctx: "ScriptedContext", *, leave_refs: int) -> None:
    """Consume lock-ref capacity until only ``leave_refs`` remain free.

    Lock-ref analogue of :meth:`exhaust_kv` — exercises the lock-ref
    exhaustion branch without needing a deeply-shared radix tree.

    Consumed by: test_lock_ref_exhaustion_blocks_admit (kv_pressure).
    """
    raise NotImplementedError(
        "scripted_runtime: exhaust_lock_refs is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def row_pool_used(ctx: "ScriptedContext") -> int:
    """Return current row-pool occupancy (number of used rows).

    Read-only counter useful for invariant assertions like
    "row pool occupancy returns to baseline after all reqs finish".

    Consumed by: test_row_pool_occupancy_returns_to_baseline (invariants).
    """
    raise NotImplementedError(
        "scripted_runtime: row_pool_used is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )
