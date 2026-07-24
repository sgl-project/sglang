"""Runtime prefill<->decode role switching for PD disaggregation.

The token KV pool is role-independent and never reallocated; only the
role-specific disaggregation structures are torn down and rebuilt on a flip.
Kept out of scheduler.py to avoid growing it further.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import PdRoleSwitchReqInput, PdRoleSwitchReqOutput

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def handle_pd_role_switch(
    scheduler: Scheduler, recv_req: PdRoleSwitchReqInput
) -> PdRoleSwitchReqOutput:
    """Flip the scheduler's disaggregation role at runtime. The instance must be
    idle; rebuild failure is fatal to the instance (no in-place rollback)."""
    old_role = scheduler.disaggregation_mode.value
    new_role = (recv_req.new_role or "").lower()

    def _fail(msg: str) -> PdRoleSwitchReqOutput:
        logger.warning(
            "PD role switch rejected (%s -> %s): %s", old_role, new_role, msg
        )
        return PdRoleSwitchReqOutput(
            success=False, message=msg, old_role=old_role, new_role=new_role
        )

    reason = _reject_reason(scheduler, new_role)
    if reason is not None:
        return _fail(reason)
    if new_role == old_role:
        return PdRoleSwitchReqOutput(
            success=True,
            message="already in target role",
            old_role=old_role,
            new_role=new_role,
        )
    if not scheduler.is_fully_idle():
        return _fail("instance is not idle; drain all requests before switching")

    scheduler._pd_role_switch_in_progress = True
    try:
        scheduler._teardown_disaggregation()
        scheduler.server_args.disaggregation_mode = new_role
        try:
            scheduler.init_disaggregation()
        except Exception as e:
            # No in-place rollback; a half-rebuilt instance isn't safe to serve.
            scheduler._pd_role_switch_unhealthy = True
            logger.critical(
                "PD role switch rebuild (%s -> %s) failed; instance unhealthy: %s",
                old_role,
                new_role,
                e,
            )
            return _fail(f"rebuild failed; instance unhealthy, restart required: {e}")

        if new_role == "decode":
            # Best-effort deferred capture; a failure only degrades to eager.
            try:
                scheduler.tp_worker.ensure_decode_cuda_graphs(
                    recv_req.decode_cuda_graph_bs
                )
            except Exception:
                logger.exception("Decode CUDA graph capture on role switch failed")

        # Break out of the old-role event loop so the supervisor re-dispatches.
        scheduler._event_loop_should_restart = True
        logger.info("PD role switch succeeded: %s -> %s", old_role, new_role)
        return PdRoleSwitchReqOutput(
            success=True, message="ok", old_role=old_role, new_role=new_role
        )
    except Exception as e:
        logger.exception("PD role switch failed")
        return _fail(f"role switch raised: {e}")
    finally:
        scheduler._pd_role_switch_in_progress = False


def _reject_reason(scheduler: Scheduler, new_role: str) -> Optional[str]:
    """Why the switch must be rejected before draining, or None to proceed.

    Table-driven: the first failing precondition's message is returned.
    """
    sa = scheduler.server_args
    km = _current_kv_manager(scheduler)
    # (failed?, lazy message). Messages are callables so only the selected one
    # is built (avoids touching fields irrelevant to the failing check).
    checks = (
        (
            not sa.enable_pd_role_switch,
            lambda: "--enable-pd-role-switch is not set on this instance",
        ),
        (
            scheduler._pd_role_switch_unhealthy,
            lambda: "instance is unhealthy after a failed role switch; restart required",
        ),
        (
            scheduler._pd_role_switch_in_progress,
            lambda: "another role switch is already in progress",
        ),
        (
            new_role not in ("prefill", "decode"),
            lambda: f"invalid new_role={new_role!r}",
        ),
        (
            scheduler.disaggregation_mode == DisaggregationMode.NULL,
            lambda: "instance is not running in PD disaggregation mode",
        ),
        (
            km is not None and not km.supports_role_switch,
            lambda: f"transfer backend {sa.disaggregation_transfer_backend!r} "
            "does not support runtime role switch",
        ),
    )
    return next((msg() for failed, msg in checks if failed), None)


def _current_kv_manager(scheduler: Scheduler):
    """The KV manager of the current role's disaggregation queue, or None."""
    if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        q = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
    elif scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        q = getattr(scheduler, "disagg_decode_prealloc_queue", None)
    else:
        q = None
    return getattr(q, "kv_manager", None) if q is not None else None


def teardown_disaggregation(scheduler: Scheduler) -> None:
    """Release the current role's disaggregation structures (queues, metadata
    buffers, KV transfer manager) so the other role can be rebuilt."""
    mode = scheduler.disaggregation_mode
    if mode == DisaggregationMode.PREFILL:
        q = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
        if q is not None:
            km = getattr(q, "kv_manager", None)
            if km is not None:
                km.teardown()
            scheduler.disagg_prefill_bootstrap_queue = None
        scheduler.disagg_prefill_inflight_queue = []
    elif mode == DisaggregationMode.DECODE:
        q = getattr(scheduler, "disagg_decode_prealloc_queue", None)
        if q is not None:
            km = getattr(q, "kv_manager", None)
            if km is not None:
                km.teardown()
            scheduler.disagg_decode_prealloc_queue = None
        scheduler.disagg_decode_transfer_queue = None
    scheduler.disagg_metadata_buffers = None
    scheduler.req_to_metadata_buffer_idx_allocator = None
    _release_prefix_cache_for_role_switch(scheduler)


def _release_prefix_cache_for_role_switch(scheduler: Scheduler) -> None:
    """Release the prefix (radix/hicache) cache so a flip works with radix ON.

    With radix disabled (ChunkCache) the flip needs nothing here: ChunkCache
    keeps no persistent prefixes and, since the instance is idle before the
    switch, the allocator is already empty. This is the historical
    ``--disable-radix-cache`` path, left untouched by the guard below.

    With radix (or hicache) enabled, finished prefixes stay in the tree and keep
    their KV-pool slots *locked* even while idle. Carried across a role switch
    that means (a) the new role would match against stale prefixes whose KV no
    longer means what it did (corruption) and (b) those locked slots would leak
    on every flip. Reset mirrors ``Scheduler.flush_cache``'s cache-release block
    (the instance is already fully idle, checked before teardown) and, for
    hicache, best-effort clears the storage backend so it is released completely.
    """
    if scheduler.disable_radix_cache:
        return
    tree_cache = scheduler.tree_cache
    if tree_cache is not None:
        clear_storage = getattr(tree_cache, "clear_storage_backend", None)
        if callable(clear_storage):
            try:
                clear_storage()
            except Exception:
                logger.exception("hicache storage release on role switch failed")
        tree_cache.reset()
    scheduler.req_to_token_pool.clear()
    scheduler.token_to_kv_pool_allocator.clear()
