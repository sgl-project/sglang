from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple

import zmq

from sglang.srt.environ import envs
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
from sglang.test.scripted_runtime.utils import (
    close_zmq_socket,
    ensure_script_importable,
    resolve_fn,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )

logger = logging.getLogger(__name__)

RESET_DRAIN_MAX_STEPS: int = 200


@dataclass(frozen=True, slots=True)
class ScriptedBatchRecord:
    forward_iter: int
    mode: Optional[str]
    rids: Tuple[str, ...]
    extend_rids: Tuple[str, ...]
    chunked_rid: Optional[str]


def _reset_engine_state(ctx: ScriptedContext) -> Generator:
    scheduler = ctx.scheduler

    ctx.abort_all()
    for _ in range(RESET_DRAIN_MAX_STEPS):
        yield
        if (
            scheduler.chunked_req is None
            and len(scheduler.waiting_queue) == 0
            and scheduler.running_batch.is_empty()
        ):
            break

    server_args = scheduler.server_args
    for _ in range(2 * (server_args.pp_size + server_args.pp_async_batch_depth)):
        yield

    ctx.flush_cache()
    yield


class ScriptedSchedulerHook:

    def __init__(
        self,
        *,
        scheduler: "Scheduler",
        tokenizer_recv_proxy: Optional["ScriptedTokenizerRecvProxy"],
    ) -> None:
        self.scheduler = scheduler
        self._is_driver = (
            scheduler.ps.pp_rank == 0
            and scheduler.ps.tp_rank == 0
            and scheduler.ps.attn_cp_rank == 0
        )
        self._batch_log: List["ScriptedBatchRecord"] = []

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
                        yield from _reset_engine_state(ctx)
                        self._batch_log.clear()
                        sub_gen = fn(ctx, *args)
                        try:
                            yield from sub_gen
                        except Exception:
                            socket.send_pyobj(
                                ScriptFailed(traceback=traceback.format_exc())
                            )
                        else:
                            socket.send_pyobj(ScriptSucceeded())
                    case _:
                        raise ValueError(f"dispatch loop: unknown command {msg!r}")
        finally:
            close_zmq_socket(socket, ctx_zmq)
            self._http_poster.close()

    def on_run_batch(self, batch) -> None:
        if not self._is_driver:
            return
        chunked = self.scheduler.chunked_req
        self._batch_log.append(
            ScriptedBatchRecord(
                forward_iter=batch.forward_iter,
                mode=(
                    batch.forward_mode.name.lower()
                    if batch.forward_mode is not None
                    else None
                ),
                rids=tuple(r.rid for r in batch.reqs),
                extend_rids=(
                    tuple(r.rid for r in batch.reqs)
                    if batch.forward_mode is not None and batch.forward_mode.is_extend()
                    else ()
                ),
                chunked_rid=chunked.rid if chunked is not None else None,
            )
        )

    def step(self) -> None:
        if not self._is_driver:
            return

        done, exc_tb = _advance_generator(self._script_fn_generator)
        if not done:
            return

        if exc_tb is not None:
            _write_out_of_band_error(exc_tb)
        sys.exit(0 if exc_tb is None else 1)


def _write_out_of_band_error(exc_tb: str) -> None:
    path = envs.SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH.get()
    if not path:
        return
    error = OutOfBandError(traceback=exc_tb or "<no traceback>")
    try:
        Path(path).write_text(error.to_json())
    except OSError:
        logger.exception(
            "Failed to write scripted_runtime out-of-band error to %s", path
        )


def _advance_generator(generator: Generator) -> Tuple[bool, Optional[str]]:
    try:
        next(generator)
        return False, None
    except StopIteration:
        return True, None
    except Exception:
        logger.exception("Failed to advance generator")
        return True, traceback.format_exc()
