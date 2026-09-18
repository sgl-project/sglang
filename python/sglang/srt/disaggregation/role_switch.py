"""Runtime prefill<->decode role switching for PD disaggregation.

The token KV pool is role-independent and never reallocated; only the
role-specific disaggregation structures are torn down and rebuilt on a flip.
Kept out of scheduler.py to avoid growing it further.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import PdRoleSwitchReqInput, PdRoleSwitchReqOutput
from sglang.srt.runtime_context import get_context, get_disagg
from sglang.srt.utils import get_available_gpu_memory

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class PdRoleSwitchRestart(Exception):
    """Break out of the current role's event loop after a successful switch."""


def run_event_loop_supervisor(
    scheduler: Scheduler, dispatch_once: Callable[[Scheduler], None]
) -> None:
    """Re-dispatch the scheduler event loop after each runtime role switch."""
    while True:
        try:
            return dispatch_once(scheduler)
        except PdRoleSwitchRestart:
            logger.info(
                "Re-dispatching event loop after PD role switch -> %s",
                scheduler.disaggregation_mode.value,
            )


def handle_pd_role_switch(
    scheduler: Scheduler, recv_req: PdRoleSwitchReqInput
) -> PdRoleSwitchReqOutput:
    """Flip the scheduler's disaggregation role at runtime. The instance must be
    idle; rebuild failure is fatal to the instance (no in-place rollback)."""
    old_role = scheduler.disaggregation_mode.value
    new_role = (recv_req.new_role or "").lower()

    def _fail(msg: str, safe_to_restore: bool = False) -> PdRoleSwitchReqOutput:
        logger.warning(
            "PD role switch rejected (%s -> %s): %s", old_role, new_role, msg
        )
        return PdRoleSwitchReqOutput(
            success=False,
            message=msg,
            old_role=old_role,
            new_role=new_role,
            safe_to_restore=safe_to_restore,
        )

    rejection = _reject_reason(scheduler, new_role)
    if rejection is not None:
        return _fail(*rejection)
    if new_role == old_role:
        return PdRoleSwitchReqOutput(
            success=True,
            message="already in target role",
            old_role=old_role,
            new_role=new_role,
        )
    if not scheduler.is_fully_idle():
        return _fail(
            "instance is not idle; drain all requests before switching",
            safe_to_restore=True,
        )

    required_graph_gb = recv_req.decode_cuda_graph_memory_gb
    # Same condition ensure_decode_cuda_graphs skips on, so the check cannot be
    # bypassed while the capture still runs.
    will_capture_graphs = (
        new_role == "decode" and not scheduler.tp_worker.get_decode_cuda_graph_bs()
    )
    if will_capture_graphs and required_graph_gb is None:
        return _fail(
            "decode_cuda_graph_memory_gb is required before capturing decode graphs",
            safe_to_restore=True,
        )
    if will_capture_graphs and required_graph_gb is not None:
        if required_graph_gb < 0:
            return _fail(
                "decode_cuda_graph_memory_gb must be non-negative",
                safe_to_restore=True,
            )
        try:
            available_graph_gb = get_available_gpu_memory(
                scheduler.device, scheduler.ps.gpu_id
            )
        except Exception as e:
            return _fail(
                f"failed to check decode CUDA graph headroom: {e}",
                safe_to_restore=True,
            )
        if available_graph_gb < required_graph_gb:
            return _fail(
                "insufficient decode CUDA graph headroom: "
                f"required={required_graph_gb:.3f} GB, "
                f"available={available_graph_gb:.3f} GB",
                safe_to_restore=True,
            )

    scheduler._pd_role_switch_in_progress = True
    try:
        # Teardown + role flip + rebuild are one logical atomic step. If any of
        # them raises, the instance is left half-torn-down (old role released,
        # new role not up) and isn't safe to serve, so mark it unhealthy. There
        # is no in-place rollback.
        try:
            teardown_disaggregation(scheduler)
            get_context().override("role_switch.flip", disaggregation_mode=new_role)
            scheduler.init_disaggregation()
            scheduler._sync_disaggregation_mode_to_subcomponents()
        except Exception as e:
            scheduler._pd_role_switch_unhealthy = True
            logger.critical(
                "PD role switch (%s -> %s) failed during teardown/rebuild; "
                "instance unhealthy: %s",
                old_role,
                new_role,
                e,
            )
            return _fail(
                f"role switch failed; instance unhealthy, restart required: {e}"
            )

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


def _reject_reason(scheduler: Scheduler, new_role: str) -> Optional[Tuple[str, bool]]:
    """Why the switch must be rejected before draining, or None to proceed.

    Table-driven: the first failing precondition's message is returned.
    """
    sa = scheduler.server_args
    km = _current_kv_manager(scheduler)
    # (failed?, safe to restore routing?, lazy message)
    checks = (
        (
            not sa.enable_pd_role_switch,
            True,
            lambda: "--enable-pd-role-switch is not set on this instance",
        ),
        (
            scheduler._pd_role_switch_unhealthy,
            False,
            lambda: (
                "instance is unhealthy after a failed role switch; restart required"
            ),
        ),
        (
            scheduler._pd_role_switch_in_progress,
            False,
            lambda: "another role switch is already in progress",
        ),
        (
            new_role not in ("prefill", "decode"),
            True,
            lambda: f"invalid new_role={new_role!r}",
        ),
        (
            scheduler.disaggregation_mode == DisaggregationMode.NULL,
            True,
            lambda: "instance is not running in PD disaggregation mode",
        ),
        (
            km is not None and not km.supports_role_switch,
            True,
            lambda: (
                f"transfer backend {get_disagg().disaggregation_transfer_backend!r} "
                "does not support runtime role switch"
            ),
        ),
        (
            getattr(km, "enable_staging", False),
            True,
            lambda: (
                "staging buffer (SGLANG_DISAGG_STAGING_BUFFER) is not "
                "supported with runtime role switch"
            ),
        ),
    )
    return next(
        ((msg(), safe_to_restore) for failed, safe_to_restore, msg in checks if failed),
        None,
    )


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
        # clear socket ctx in CommonKVReceiver
        CommonKVReceiver.close_all_sockets()
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
