"""Free functions for read-only scheduler state queries.

These expose batch composition, queue contents, forward mode, engine
stats, load-inquirer counters, scheduler-path observability, and
DP-attention per-rank state. Each takes the facade ``ctx`` first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def chunked_in_flight_count(ctx: "ScriptedContext") -> int:
    """Return the number of requests currently mid-chunked-prefill.

    A request is counted while it has at least one finished chunk
    but has not yet completed its prefill.

    Consumed by: test_chunked_in_flight_invariant (invariants),
                 test_concurrent_chunked_reqs (multi_req).
    """
    raise NotImplementedError(
        "scripted_runtime: chunked_in_flight_count is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def get_chunked_req_rid(ctx: "ScriptedContext") -> Optional[str]:
    """Return the rid currently held in ``Scheduler.chunked_req``, or None.

    Reflects the singular "current chunked req" slot in the scheduler;
    useful for special_case assertions about which req owns the slot.

    Consumed by: test_chunked_req_slot_ownership (special_case).
    """
    raise NotImplementedError(
        "scripted_runtime: get_chunked_req_rid is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def is_idle(ctx: "ScriptedContext") -> bool:
    """Return True if the engine reported IDLE on the current iter.

    Single-iter snapshot — does not imply no waiting/chunked work.

    Consumed by: test_engine_idle_between_reqs (invariants).
    """
    raise NotImplementedError(
        "scripted_runtime: is_idle is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def is_fully_idle(ctx: "ScriptedContext") -> bool:
    """Return True if engine is idle, has no chunked in-flight, and waiting queue is empty.

    Stronger than :meth:`is_idle` — useful as a quiescence gate
    between phases of a test.

    Consumed by: test_engine_fully_idle_after_drain (invariants),
                 test_chunked_req_slot_ownership (special_case).
    """
    raise NotImplementedError(
        "scripted_runtime: is_fully_idle is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def batch_size(ctx: "ScriptedContext") -> int:
    """Return current ``running_batch.size()``.

    Consumed by: test_batch_size_under_kv_pressure (regression),
                 test_batch_size_after_retract (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: batch_size is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def batch_composition(ctx: "ScriptedContext") -> Dict[str, List[str]]:
    """Return a breakdown of current batch by forward-mode role.

    Shape: ``{"prefill": [...rids], "decode": [...rids], "chunked": [...rids]}``.

    Consumed by: test_batch_composition_chunked_plus_decode (multi_req),
                 test_batch_composition_invariants (invariants).
    """
    raise NotImplementedError(
        "scripted_runtime: batch_composition is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def batch_rids(ctx: "ScriptedContext") -> Set[str]:
    """Return the set of rids currently in the batch (PP cross-mb dedup applied).

    For PP, deduplicates across micro-batches so a req in flight on
    both mbs only appears once.

    Consumed by: test_pp_cross_mb_dedup (pp),
                 test_batch_rids_invariant (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: batch_rids is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def waiting_rids(ctx: "ScriptedContext") -> Set[str]:
    """Return the set of rids currently in ``waiting_queue``.

    Consumed by: test_waiting_rids_after_kv_pressure (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: waiting_rids is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def running_rids(ctx: "ScriptedContext") -> Set[str]:
    """Return the set of rids currently in ``running_batch``.

    Consumed by: test_running_rids_after_retract (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: running_rids is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def in_flight_other_mb_rids(ctx: "ScriptedContext") -> Set[str]:
    """Return rids in micro-batches other than the current iter's mb.

    Non-empty only under PP (``pp_size > 1``); returns empty set
    otherwise.

    Consumed by: test_pp_in_flight_other_mb_visible (pp),
                 test_pp_cross_mb_dedup (pp).
    """
    raise NotImplementedError(
        "scripted_runtime: in_flight_other_mb_rids is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def list_active_reqs(ctx: "ScriptedContext") -> List[str]:
    """Return rids of all requests the engine still owns.

    Includes waiting + running + chunked + cross-mb. Useful for
    leak-detection invariants.

    Consumed by: test_no_req_leaks_after_drain (invariants).
    """
    raise NotImplementedError(
        "scripted_runtime: list_active_reqs is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def forward_mode(ctx: "ScriptedContext") -> str:
    """Return current ``ForwardMode`` name (e.g. "EXTEND", "DECODE", "MIXED", "IDLE").

    Consumed by: test_forward_mode_transitions (invariants).
    """
    raise NotImplementedError(
        "scripted_runtime: forward_mode is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def engine_stats(ctx: "ScriptedContext") -> Dict[str, Any]:
    """Return a dict of internal engine counters and snapshots.

    Open-ended bag (radix hit_count, kv pool_free, mem stats, etc.).
    Specific keys are documented as they land.

    Consumed by: test_engine_stats_snapshot (invariants),
                 test_radix_hit_count_increments (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: engine_stats is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def kv_pool_underflow_count(ctx: "ScriptedContext") -> int:
    """Return the counter for "release_kv_cache called with token count > committed" near-misses.

    Pre-fix, the abort dual-queue bug would bump this counter; the
    invariant is "count stays at 0".

    Consumed by: test_no_kv_pool_underflow_under_abort (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: kv_pool_underflow_count is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def lock_refs_snapshot(ctx: "ScriptedContext") -> int:
    """Return the total ``lock_ref`` count summed across all radix nodes.

    Useful as a leak-detection invariant: should return to 0 once
    all reqs finish.

    Consumed by: test_lock_refs_return_to_zero (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: lock_refs_snapshot is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def load_inquirer_num_pending_tokens(ctx: "ScriptedContext") -> int:
    """Return ``LoadInquirer._get_num_pending_tokens()``.

    Used by router/dispatcher logic; tests verify it tracks the
    actual pending workload.

    Consumed by: test_load_inquirer_pending_tokens (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: load_inquirer_num_pending_tokens is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def load_inquirer_snapshot(ctx: "ScriptedContext") -> Dict[str, int]:
    """Return the full LoadInquirer dict snapshot.

    Open-ended dict mirroring the load-inquirer fields.

    Consumed by: test_load_inquirer_snapshot (invariants).
    """
    raise NotImplementedError(
        "scripted_runtime: load_inquirer_snapshot is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def last_admission_path(ctx: "ScriptedContext") -> Optional[str]:
    """Return which admit branch fired on the last iter.

    One of: "new", "reuse", "chunked_resume", "tree_cache_resume", or
    ``None`` if no admit happened.

    Consumed by: test_admission_path_chunked_resume (special_case),
                 test_admission_path_tree_cache_resume (special_case).
    """
    raise NotImplementedError(
        "scripted_runtime: last_admission_path is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def last_scheduler_path(ctx: "ScriptedContext") -> Optional[str]:
    """Return which top-level branch was taken in ``get_next_batch_to_run``.

    One of: "idle", "stash", "merge", "admit", ... (open-ended); ``None``
    if no scheduling decision was made on the last iter.

    Consumed by: test_scheduler_path_stash_then_merge (regression).
    """
    raise NotImplementedError(
        "scripted_runtime: last_scheduler_path is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def last_chunked_exclude_set_source(ctx: "ScriptedContext") -> Optional[str]:
    """Return where the chunked_req_to_exclude set came from on the last iter.

    One of: "chunked_req", "last_batch.reqs", or ``None``.

    Consumed by: test_chunked_exclude_set_source (special_case).
    """
    raise NotImplementedError(
        "scripted_runtime: last_chunked_exclude_set_source is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def dp_rank_max_pending(ctx: "ScriptedContext", rank: int) -> int:
    """Return max pending tokens at the given DP rank.

    Only meaningful when ``dp_size > 1``.

    Consumed by: test_dp_rank_max_pending_balanced (dp_attention).
    """
    raise NotImplementedError(
        "scripted_runtime: dp_rank_max_pending is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )


def dp_rank_is_idle(ctx: "ScriptedContext", rank: int) -> bool:
    """Return True if the given DP rank is fully idle.

    Per-rank version of :meth:`is_fully_idle`. Only meaningful when
    ``dp_size > 1``.

    Consumed by: test_dp_rank_idle_invariant (dp_attention).
    """
    raise NotImplementedError(
        "scripted_runtime: dp_rank_is_idle is wishlist — see "
        "2026-05-26-round-5-de-skip-and-api-wishlist.md"
    )
