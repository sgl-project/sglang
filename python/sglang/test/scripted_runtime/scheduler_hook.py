from __future__ import annotations

import logging
import sys
import time
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
# Below the test-side LISTENER_ACCEPT_TIMEOUT_S so a stuck warmup surfaces as
# this specific error instead of a generic handshake timeout.
WARMUP_DRIVE_TIMEOUT_S: float = 120.0


@dataclass(frozen=True, slots=True)
class ScriptedBatchRecord:
    forward_iter: int
    mode: Optional[str]
    rids: Tuple[str, ...]
    extend_rids: Tuple[str, ...]
    chunked_rid: Optional[str]


def _drive_engine_through_warmup(ctx: ScriptedContext) -> Generator:
    """Run the engine until the server warmup request has been received and
    fully processed, so scripts never observe foreign warmup traffic."""
    scheduler = ctx.scheduler
    server_args = scheduler.server_args
    if server_args.skip_server_warmup:
        logger.info("scripted_runtime: skip_server_warmup set, not driving warmup")
        return

    logger.info("scripted_runtime: driving engine until server warmup completes")
    start_time = time.monotonic()

    # is_fully_idle() can transiently report idle while a PP microbatch result
    # is still in flight, so require it to hold for two full microbatch
    # rotations after the warmup request was observed on the recv socket.
    quiesce_iters = 2 * (server_args.pp_size + server_args.pp_async_batch_depth)
    proxy = ctx._tokenizer_recv_proxy
    deadline = start_time + WARMUP_DRIVE_TIMEOUT_S

    idle_streak = 0
    while idle_streak < quiesce_iters:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "scripted_runtime: server warmup did not complete within "
                f"{WARMUP_DRIVE_TIMEOUT_S}s "
                f"(work_reqs_seen={proxy.work_reqs_seen}, "
                f"idle_streak={idle_streak})"
            )
        yield
        if proxy.work_reqs_seen > 0 and scheduler.is_fully_idle():
            idle_streak += 1
        else:
            idle_streak = 0

    logger.info(
        "scripted_runtime: server warmup drained in %.1fs (work_reqs_seen=%d)",
        time.monotonic() - start_time,
        proxy.work_reqs_seen,
    )


def _reset_engine_state(ctx: ScriptedContext) -> Generator:
    scheduler = ctx.scheduler

    if scheduler._engine_paused:
        ctx.continue_generation()

    ctx._release_exhausted_pools()
    ctx.abort_all()
    for _ in range(RESET_DRAIN_MAX_STEPS):
        yield
        if scheduler.is_fully_idle():
            break
    else:
        raise RuntimeError(
            "scripted_runtime reset: scheduler did not become fully idle "
            f"within {RESET_DRAIN_MAX_STEPS} steps"
        )

    ctx.flush_cache()
    yield


class ScriptedSchedulerHook:

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        tokenizer_recv_proxy: Optional[ScriptedTokenizerRecvProxy],
    ) -> None:
        self.scheduler = scheduler
        self._is_driver = (
            scheduler.ps.pp_rank == 0
            and scheduler.ps.tp_rank == 0
            and scheduler.ps.attn_cp_rank == 0
        )
        self._batch_log: List[ScriptedBatchRecord] = []

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
            yield from _drive_engine_through_warmup(self._context)
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
