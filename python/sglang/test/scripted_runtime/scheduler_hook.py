"""ScriptedSchedulerHook: the scheduler-side half of the scripted runtime.

Constructed by ``Scheduler.__init__`` when ``scripted_runtime_fn_path`` is
set and held as a scheduler field. ``SchedulerRequestReceiver.recv_requests``
calls :meth:`step` once per event-loop iteration. On the driver rank
(``pp_rank == tp_rank == attn_cp_rank == 0``) the hook owns the script
generator and the :class:`ScriptedContext` handed to it, advancing the
generator one step per call; non-driver ranks join the cross-rank cpu
broadcast that carries the script's done / error state so every rank
``sys.exit``s together when the script finishes.

This object owns everything the *scheduler* touches (generator stepping,
cross-rank broadcast, per-req lookups for :class:`ScriptedReqHandle`); the
script-facing verbs live on :class:`ScriptedContext`.
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import TYPE_CHECKING, Any, Generator, List, Optional, Tuple

from sglang.srt.utils.common import broadcast_pyobj
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.utils import ensure_script_importable, resolve_fn

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.test.scripted_runtime.req_handle import ScriptedReqStatus
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )

logger = logging.getLogger(__name__)


class ScriptedSchedulerHook:
    """Generator-driven scheduler-side hook installed in every scheduler subprocess.

    On the driver rank it builds the :class:`ScriptedContext`, instantiates
    the script generator over it, and advances it one step per :meth:`step`
    call. When the generator finishes, every rank ``sys.exit``s so all
    scheduler subprocesses tear down together.
    """

    def __init__(
        self,
        *,
        scheduler: "Scheduler",
        script_fn_path: str,
        tokenizer_recv_proxy: Optional["ScriptedTokenizerRecvProxy"],
    ) -> None:
        self._scheduler = scheduler
        self._is_driver = (
            scheduler.ps.pp_rank == 0
            and scheduler.ps.tp_rank == 0
            and scheduler.ps.attn_cp_rank == 0
        )
        self._script_fn_path = script_fn_path

        if self._is_driver:
            ensure_script_importable(
                scheduler.server_args.scripted_runtime_sys_path_entry
            )
            script_fn = resolve_fn(script_fn_path)
            self._context: Optional[ScriptedContext] = ScriptedContext(
                scheduler_hook=self,
                tokenizer_recv_proxy=tokenizer_recv_proxy,
            )
            generator = script_fn(self._context)
            if not hasattr(generator, "__next__"):
                raise TypeError(
                    f"scripted_runtime function {script_fn_path!r} must be a "
                    f"generator (use 'yield' inside it); got {type(generator).__name__}"
                )
            self._generator: Optional[Generator] = generator
        else:
            self._context = None
            self._generator = None

    # ============================================================
    # Lookups used by ScriptedReqHandle (driver-rank-local view).
    # ============================================================
    def _lookup_req_status(self, rid: str) -> "ScriptedReqStatus":
        # TODO(reimplement): the previous implementation was wrong. It never
        # reported "finished", ignored the chunked_req slot, and under PP only
        # inspected the current microbatch's running_batch — so a req running
        # in another microbatch read as "unknown". A correct version must fold
        # in finished / chunked / cross-microbatch state before callers can
        # trust it; until then use the narrower observables (is_chunking,
        # finished, _find_req_by_rid + waiting_queue) instead.
        raise NotImplementedError(
            "scripted_runtime: _lookup_req_status needs reimplementation — see "
            "the chunked / PP / finished caveats in the comment above"
        )

    def _find_req_by_rid(self, rid: str) -> Optional[Any]:
        """Locate the raw ``Req`` by rid across scheduler queues / batches.

        Returns ``None`` if the rid is not currently held by the scheduler.
        Used by ScriptedReqHandle properties that read per-req scheduler-side state.
        """
        s = self._scheduler
        chunked = s.chunked_req
        if chunked is not None and chunked.rid == rid:
            return chunked
        for r in s.waiting_queue:
            if r.rid == rid:
                return r
        if s.running_batch is not None:
            for r in s.running_batch.reqs:
                if r.rid == rid:
                    return r
        last_batch = getattr(s, "last_batch", None)
        if last_batch is not None:
            for r in last_batch.reqs:
                if r.rid == rid:
                    return r
        return None

    def _lookup_finished(self, rid: str) -> bool:
        """True iff the req has reached a finished state (or already gone).

        Once a req is filtered out of all scheduler structures, the lookup
        returns ``True`` — by that point the req can only have left because
        it finished.
        """
        req = self._find_req_by_rid(rid)
        if req is None:
            # Req has left every scheduler structure → it finished (or was
            # aborted, which is also a finish). True is the only sensible
            # answer for a req that's no longer tracked.
            return rid in self._context._req_handles
        return req.finished()

    def _lookup_is_chunking(self, rid: str) -> bool:
        """True iff this rid is the scheduler's current chunked_req."""
        s = self._scheduler
        return s.chunked_req is not None and s.chunked_req.rid == rid

    # ============================================================
    # Internal: invoked by SchedulerRequestReceiver at every iter.
    # ============================================================

    def step(self) -> None:
        """Advance the generator one step (driver only) and broadcast
        completion state. When the script finishes or raises, every rank
        ``sys.exit``s so all scheduler subprocesses tear down together.

        ``sys.exit`` raises ``SystemExit`` (a ``BaseException``), so it sails
        past ``run_scheduler_process``'s ``except Exception`` SIGQUIT path and
        exits the subprocess cleanly with code 0 (ok) / 1 (script failed) —
        no scripted-runtime hook in the production bootstrap.
        """
        if self._is_driver:
            payload: List = list(self._advance_generator())
        else:
            # ``broadcast_pyobj`` ignores the value on non-source ranks.
            payload = []

        payload = broadcast_pyobj(
            data=payload,
            rank=self._scheduler.world_group.rank,
            dist_group=self._scheduler.world_group.cpu_group,
            src=0,
        )
        done, exc_tb = payload[0], payload[1]
        if not done:
            return

        if exc_tb is not None and self._is_driver:
            self._write_traceback(exc_tb)
        sys.exit(0 if exc_tb is None else 1)

    def _write_traceback(self, exc_tb: str) -> None:
        """Persist a failed script's traceback for the caller to surface."""
        path = self._scheduler.server_args.scripted_runtime_traceback_path
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write(exc_tb or "<no traceback>")
        except OSError:
            logger.exception("Failed to write scripted_runtime traceback to %s", path)

    def _advance_generator(self) -> Tuple[bool, Optional[str]]:
        try:
            next(self._generator)
            return (False, None)
        except StopIteration:
            return (True, None)
        except BaseException:  # noqa: BLE001 — capture every kind of failure
            return (True, traceback.format_exc())
