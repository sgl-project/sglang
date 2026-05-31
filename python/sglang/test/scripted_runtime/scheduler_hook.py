"""ScriptedSchedulerHook: the scheduler-side half of the scripted runtime.

Constructed by ``Scheduler.__init__`` when ``SGLANG_TEST_SCRIPTED_RUNTIME``
is set and held as a scheduler field. ``SchedulerRequestReceiver.recv_requests``
calls :meth:`step` once per event-loop iteration. On the driver rank
(``pp_rank == tp_rank == attn_cp_rank == 0``) the hook owns the dispatch-loop
generator and the :class:`ScriptedContext` handed to it, advancing the
generator one step per call; non-driver ranks join the cross-rank cpu
broadcast that carries the script's done / error state so every rank
``sys.exit``s together when the script finishes.

This object owns everything the *scheduler* touches: the ZMQ control plane
(the dispatch loop that pulls and runs each caller-requested sub-script),
generator stepping, cross-rank broadcast, and per-req lookups for
:class:`ScriptedReqHandle`. The script-facing verbs live on
:class:`ScriptedContext`.
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple

import zmq

from sglang.srt.environ import envs
from sglang.srt.utils.common import broadcast_pyobj
from sglang.srt.utils.network import get_zmq_socket
from sglang.test.scripted_runtime.background_http_poster import BackgroundHttpPoster
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.io_struct import (
    HookReady,
    OutOfBandError,
    RunScript,
    ScriptFailed,
    ScriptSucceeded,
    Shutdown,
)
from sglang.test.scripted_runtime.utils import ensure_script_importable, resolve_fn

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )

logger = logging.getLogger(__name__)


class ScriptedSchedulerHook:
    """Generator-driven scheduler-side hook installed in every scheduler subprocess.

    On the driver rank it builds the :class:`ScriptedContext`, opens the ZMQ
    dispatch loop over it, and advances that loop one step per :meth:`step`
    call. When the loop finishes (the caller sent :class:`Shutdown`), every
    rank ``sys.exit``s so all scheduler subprocesses tear down together.
    """

    def __init__(
        self,
        *,
        scheduler: "Scheduler",
        tokenizer_recv_proxy: Optional["ScriptedTokenizerRecvProxy"],
    ) -> None:
        self._scheduler = scheduler
        self._is_driver = (
            scheduler.ps.pp_rank == 0
            and scheduler.ps.tp_rank == 0
            and scheduler.ps.attn_cp_rank == 0
        )

        if self._is_driver:
            ensure_script_importable(
                envs.SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY.get()
            )
            self._http_poster: Optional[BackgroundHttpPoster] = BackgroundHttpPoster()
            self._context: Optional[ScriptedContext] = ScriptedContext(
                scheduler_hook=self,
                tokenizer_recv_proxy=tokenizer_recv_proxy,
                http_poster=self._http_poster,
            )
            self._script_fn_generator: Optional[Generator] = self._run_dispatch_loop()
        else:
            self._http_poster = None
            self._context = None
            self._script_fn_generator = None

    def _run_dispatch_loop(self) -> Generator:
        """Receive :class:`RunScript` / :class:`Shutdown`; ``yield from`` each sub-script.

        Runs forever on the scheduler driver rank until the caller sends a
        :class:`Shutdown`, at which point the generator returns normally and
        the scheduler tears down.

        Crucially, when a sub-script raises (including ``AssertionError``), the
        loop *captures* the traceback into a socket message and *keeps
        running* — it does not re-raise. Re-raising would tear the engine down
        (the hook ``sys.exit``s the scheduler subprocess), voiding every
        remaining test in the class.
        """
        endpoint = envs.SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR.get()
        ctx_zmq = zmq.Context()
        socket = get_zmq_socket(ctx_zmq, zmq.PAIR, endpoint, bind=False)
        try:
            socket.send_pyobj(HookReady())
            while True:
                msg = socket.recv_pyobj()
                match msg:
                    case Shutdown():
                        return
                    case RunScript(fn_path=fn_path, args=args):
                        fn = resolve_fn(fn_path)
                        ctx = self._context
                        ctx.flush_cache()
                        yield
                        sub_gen = fn(ctx, *args)
                        try:
                            yield from sub_gen
                        except BaseException:
                            socket.send_pyobj(
                                ScriptFailed(traceback=traceback.format_exc())
                            )
                        else:
                            socket.send_pyobj(ScriptSucceeded())
                    case _:
                        raise ValueError(f"dispatch loop: unknown command {msg!r}")
        finally:
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
            ctx_zmq.term()
            self._http_poster.close()

    def _find_req_by_rid(self, rid: str) -> Optional["Req"]:
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
            return rid in self._context._req_handles
        return req.finished()

    def _lookup_is_chunking(self, rid: str) -> bool:
        """True iff this rid is the scheduler's current chunked_req."""
        s = self._scheduler
        return s.chunked_req is not None and s.chunked_req.rid == rid

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
            self._write_out_of_band_error(exc_tb)
        sys.exit(0 if exc_tb is None else 1)

    def _write_out_of_band_error(self, exc_tb: str) -> None:
        """Persist a fatal scheduler-side error as JSON for the caller to surface."""
        path = envs.SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH.get()
        if not path:
            return
        error = OutOfBandError(traceback=exc_tb or "<no traceback>")
        try:
            with open(path, "w") as f:
                f.write(error.to_json())
        except OSError:
            logger.exception(
                "Failed to write scripted_runtime out-of-band error to %s", path
            )

    def _advance_generator(self) -> Tuple[bool, Optional[str]]:
        try:
            next(self._script_fn_generator)
            return (False, None)
        except StopIteration:
            return (True, None)
        except BaseException:  # noqa: BLE001 — capture every kind of failure
            return (True, traceback.format_exc())
