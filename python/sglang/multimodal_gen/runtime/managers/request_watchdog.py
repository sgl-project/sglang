# SPDX-License-Identifier: Apache-2.0
"""Request watchdog: report a request that executes past its time limit.

A hung kernel or collective blocks the workers without logging anything, and
/health only notices a worker process that exited. With ``--request-time-limit``
the driver rank of each replica reports the request it executes (and the stage
and denoising step it enters) to the HTTP server, which a hung worker cannot
block, and the HTTP server times it. Past the limit it:
  1. logs the request id, task, stage, denoising step and elapsed time;
  2. marks the server failed: /health returns 503 ``unhealthy`` and new requests
     are refused;
  3. asks every local worker to dump the Python stacks of all its threads
     (faulthandler, registered on STACK_DUMP_SIGNAL; it needs no ptrace and runs
     while the main thread is stuck in a CUDA call).
It does not abort the hung request: restart the server's process group.

The clock starts when the driver dispatches the request from its queue, so
waiting behind other requests does not count. Warmup requests are not timed.
"""

from __future__ import annotations

import asyncio
import faulthandler
import math
import os
import signal
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Iterator

import psutil
import zmq
import zmq.asyncio

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)

# SIGUSR1 is taken by the CUDA graph lifetime tracer of the tests
STACK_DUMP_SIGNAL = signal.SIGUSR2
# limit key matching every task
ANY_TASK = "*"
CHECK_INTERVAL_MS = 1000
# one worker at a time, so the dumps do not interleave in a shared log
STACK_DUMP_GAP_S = 0.25


def parse_request_time_limits(entries: list[str] | None) -> dict[str, float]:
    """``--request-time-limit`` entries to {task or ANY_TASK: seconds}.

    Each entry is SECONDS (every task) or TASK=SECONDS; 0 or ``off`` removes
    the limit. An empty result means the watchdog is off.
    """
    limits: dict[str, float] = {}
    for entry in entries or ():
        task, _, value = entry.strip().rpartition("=")
        task = task.strip().lower() or ANY_TASK
        value = value.strip().lower()
        try:
            seconds = 0.0 if value == "off" else float(value)
        except ValueError:
            seconds = math.nan
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(
                f"--request-time-limit {entry!r}: expected SECONDS or "
                "TASK=SECONDS with SECONDS >= 0 (0 or off: no limit)"
            )
        if seconds > 0:
            limits[task] = seconds
        else:
            limits.pop(task, None)
    return limits


def request_time_limit(limits: dict[str, float], task: str | None) -> float | None:
    if task is not None and task in limits:
        return limits[task]
    return limits.get(ANY_TASK)


def request_watchdog_endpoint(server_args: Any) -> str | None:
    """Where replica drivers report executions, or None when the watchdog is
    off. Like the progress endpoint, ipc only reaches this node."""
    if server_args.progress_endpoint is None or not parse_request_time_limits(
        server_args.request_time_limit
    ):
        return None
    return f"ipc://{tempfile.gettempdir()}/sglang-diffusion-watchdog-{server_args.port}"


def install_stack_dump_handler(server_args: Any) -> None:
    """Every worker: dump all threads' Python stacks to stderr on
    STACK_DUMP_SIGNAL, which the HTTP server sends when a request is overdue."""
    if request_watchdog_endpoint(server_args) is not None:
        faulthandler.register(STACK_DUMP_SIGNAL, all_threads=True)


class ExecutionReporter:
    """Replica driver side: reports the request it executes to the HTTP server.

    Messages are JSON lists ``[replica, event, *args]``: ``begin`` (request id,
    task, limit, start time on the monotonic clock this node shares), ``stage``
    or ``step`` (profiler region name) and ``end``.
    """

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        limits: dict[str, float],
        replica: int,
    ):
        self._socket = context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._limits = limits
        self._replica = replica
        # thread executing the timed request; regions entered by other threads
        # (the output finalizer) are not part of it
        self._owner: int | None = None

    def _send(self, *message: Any) -> None:
        try:
            self._socket.send_json([self._replica, *message], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    @contextmanager
    def watch(self, req: Req | None) -> Iterator[None]:
        limit = task = None
        if (
            req is not None
            and not req.is_warmup
            and not req.extra.get("server_internal_prewarm")
        ):
            if req.sampling_params is not None:
                task = req.sampling_params.request_task()
            limit = request_time_limit(self._limits, task)
        if limit is None:
            yield
            return
        self._owner = threading.get_ident()
        self._send("begin", req.request_id, task, limit, time.monotonic())
        try:
            yield
        finally:
            self._owner = None
            self._send("end")

    def enter_region(self, name: str, is_step: bool) -> None:
        if self._owner == threading.get_ident():
            self._send("step" if is_step else "stage", name)

    def watch_dispatch(self, dispatch):
        from sglang.multimodal_gen.runtime.server_warmup import get_first_generation_req

        @wraps(dispatch)
        def watched(items, *args):
            with self.watch(get_first_generation_req(items[0][1])):
                return dispatch(items, *args)

        return watched


execution_reporter: ExecutionReporter | None = None


def init_execution_reporter(
    context: zmq.Context, server_args: Any, replica: int
) -> ExecutionReporter | None:
    """Replica driver: start reporting executions if the watchdog is on."""
    global execution_reporter
    endpoint = request_watchdog_endpoint(server_args)
    if endpoint is None:
        execution_reporter = None
    else:
        execution_reporter = ExecutionReporter(
            context,
            endpoint,
            parse_request_time_limits(server_args.request_time_limit),
            replica,
        )
    return execution_reporter


@dataclass
class _Execution:
    request_id: str
    task: str | None
    limit_s: float
    started: float
    stage: str | None = None
    step: str | None = None


class RequestWatchdog:
    """HTTP server side: times each replica's executing request."""

    def __init__(
        self,
        workers: list[psutil.Process],
        mark_failed: Callable[[str], None],
    ):
        self._workers = workers
        self._mark_failed = mark_failed
        self._executing: dict[int, _Execution] = {}

    def handle(self, message: list[Any]) -> None:
        replica, event, *args = message
        if event == "begin":
            self._executing[replica] = _Execution(*args)
            return
        if event == "end":
            self._executing.pop(replica, None)
            return
        execution = self._executing.get(replica)
        if execution is None:
            return
        if event == "stage":
            execution.stage, execution.step = args[0], None
        else:
            execution.step = args[0]

    def overdue(self, now: float) -> _Execution | None:
        for execution in self._executing.values():
            if now - execution.started > execution.limit_s:
                return execution
        return None

    async def trip(self, execution: _Execution, now: float) -> None:
        error = f"request {execution.request_id} exceeded {execution.limit_s:g} s"
        logger.error(
            "Request watchdog: %s (task %s, executing for %.1f s, stage %s, "
            "denoising step %s). /health now reports unhealthy; restart the "
            "server's process group. Python stacks of the %d worker(s) follow.",
            error,
            execution.task,
            now - execution.started,
            execution.stage,
            execution.step,
            len(self._workers),
        )
        self._mark_failed(error)
        # let the requests this fails log their errors before the dumps start
        await asyncio.sleep(STACK_DUMP_GAP_S)
        for rank, worker in enumerate(self._workers):
            logger.error("Python stacks of local worker %d (pid %d):", rank, worker.pid)
            with suppress(psutil.NoSuchProcess):
                worker.send_signal(STACK_DUMP_SIGNAL)
            await asyncio.sleep(STACK_DUMP_GAP_S)

    async def run(self, endpoint: str) -> None:
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.PULL)
        try:
            try:
                socket.bind(endpoint)
            except zmq.ZMQError as e:
                logger.error("Request watchdog off: cannot bind %s: %s", endpoint, e)
                return
            while True:
                if await socket.poll(timeout=CHECK_INTERVAL_MS):
                    self.handle(await socket.recv_json())
                now = time.monotonic()
                execution = self.overdue(now)
                if execution is not None:
                    await self.trip(execution, now)
                    return
        finally:
            context.destroy(linger=0)
            with suppress(FileNotFoundError):
                os.unlink(endpoint.removeprefix("ipc://"))


def start_request_watchdog(
    server_args: Any,
    workers: list[psutil.Process],
    mark_failed: Callable[[str], None],
) -> asyncio.Task | None:
    """HTTP server lifespan: start the watchdog if ``--request-time-limit``
    sets a limit."""
    endpoint = request_watchdog_endpoint(server_args)
    if endpoint is None:
        return None
    logger.info(
        "Request watchdog on, execution time limits in seconds: %s",
        parse_request_time_limits(server_args.request_time_limit),
    )
    return asyncio.create_task(RequestWatchdog(workers, mark_failed).run(endpoint))
