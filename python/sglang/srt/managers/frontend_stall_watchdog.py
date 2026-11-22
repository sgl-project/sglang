"""Exit the server when its frontend stops making progress.

The scheduler watchdogs run inside the scheduler processes, so they cannot see
a stuck tokenizer event loop or a scheduler that stops answering while alive.
This watchdog runs in the HTTP (tokenizer) process and fires when either:

- the event loop does not run for `timeout` seconds, or
- user requests are in flight and a one-token probe gets no scheduler answer
  within `timeout` seconds.

Firing dumps diagnostics, kills the child processes, and exits the process with
a non-zero code, so a supervisor such as Kubernetes restarts the server.
"""

import asyncio
import faulthandler
import logging
import os
import sys
import threading
import time
from typing import Awaitable, Callable, Optional

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.cudacore_pyspy_dump_utils import pyspy_dump_schedulers

logger = logging.getLogger(__name__)

# Arbitrary bounds so a hung py-spy or dump cannot delay the exit indefinitely.
_DIAGNOSTICS_TIMEOUT_S = 30.0
_KILL_CHILDREN_WAIT_S = 15.0


class FrontendStallWatchdog:
    def __init__(
        self,
        *,
        tokenizer_manager,
        timeout: float,
        probe_fn: Callable[[float], Awaitable[bool]],
        probe_interval: float = 10.0,
        heartbeat_interval: float = 1.0,
        monitor_interval: float = 1.0,
        diagnostics_timeout: float = _DIAGNOSTICS_TIMEOUT_S,
        exit_fn: Callable[[int], None] = os._exit,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._tokenizer_manager = tokenizer_manager
        self._timeout = timeout
        self._probe_fn = probe_fn
        self._probe_interval = probe_interval
        self._heartbeat_interval = heartbeat_interval
        self._monitor_interval = monitor_interval
        self._diagnostics_timeout = diagnostics_timeout
        self._exit_fn = exit_fn
        self._clock = clock
        self._last_heartbeat = clock()
        # Written by the probe task, read by the monitor thread.
        self._failure: Optional[str] = None
        self._stop_event = threading.Event()
        self._tasks = []
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the loop tasks and the monitor thread; call on the serving loop."""
        loop = asyncio.get_running_loop()
        self._last_heartbeat = self._clock()
        self._tasks = [
            loop.create_task(self._heartbeat_loop()),
            loop.create_task(self._probe_loop()),
        ]
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="frontend-stall-watchdog"
        )
        self._thread.start()
        logger.info(f"Frontend stall watchdog armed (timeout={self._timeout}s).")

    def stop(self) -> None:
        self._stop_event.set()
        for task in self._tasks:
            task.cancel()

    async def _heartbeat_loop(self) -> None:
        while True:
            self._last_heartbeat = self._clock()
            await asyncio.sleep(self._heartbeat_interval)

    async def _probe_loop(self) -> None:
        while True:
            await asyncio.sleep(self._probe_interval)
            if not (self._is_monitoring() and self._user_request_count() > 0):
                continue
            answered = await self._probe_fn(self._timeout)
            # A pause, drain or weight update that began mid-probe explains silence.
            if not answered and self._is_monitoring():
                self._failure = (
                    f"no scheduler answer to a probe within {self._timeout}s "
                    f"while {self._user_request_count()} requests are in flight"
                )

    def _is_monitoring(self) -> bool:
        """Whether silence would mean a stall; requests legitimately wait otherwise."""
        tm = self._tokenizer_manager
        return (
            tm.server_status != ServerStatus.Starting
            and not tm.gracefully_exit
            and not tm.is_pause
            and not tm.model_update_lock.has_writer
        )

    def _user_request_count(self) -> int:
        return sum(
            not rid.startswith(HEALTH_CHECK_RID_PREFIX)
            for rid in list(self._tokenizer_manager.rid_to_state)
        )

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(self._monitor_interval):
            reason = self._check()
            if reason is not None:
                self._fire(reason)
                return

    def _check(self) -> Optional[str]:
        if self._failure is not None:
            return self._failure
        tm = self._tokenizer_manager
        if tm.server_status == ServerStatus.Starting or tm.gracefully_exit:
            return None
        blocked = self._clock() - self._last_heartbeat
        if blocked > self._timeout:
            return f"event loop did not run for {blocked:.0f}s"
        return None

    def _fire(self, reason: str) -> None:
        tm = self._tokenizer_manager
        tm.server_status = ServerStatus.UnHealthy
        logger.error(
            f"Frontend stall watchdog fired: {reason}. {self._describe_requests()} "
            "Dumping diagnostics, then exiting so the supervisor restarts the server."
        )
        self._run_bounded(
            "thread dump", lambda: faulthandler.dump_traceback(all_threads=True)
        )
        self._run_bounded(
            "scheduler py-spy dump", lambda: pyspy_dump_schedulers(scheduler_only=True)
        )
        self._run_bounded("request dump", tm.dump_requests_before_crash)
        # Children dying from the kill below are expected, not a crash to report.
        if tm._subprocess_watchdog is not None:
            self._run_bounded("subprocess watchdog stop", tm._subprocess_watchdog.stop)
        self._run_bounded(
            "kill child processes",
            lambda: kill_process_tree(
                os.getpid(), include_parent=False, wait_timeout=_KILL_CHILDREN_WAIT_S
            ),
        )
        _flush_logs()
        # os._exit, not sys.exit: this runs on a thread, and the process may be
        # PID 1 in a container, which ignores a SIGKILL sent from inside.
        self._exit_fn(1)

    def _describe_requests(self) -> str:
        tm = self._tokenizer_manager
        states = [
            state
            for rid, state in list(tm.rid_to_state.items())
            if not rid.startswith(HEALTH_CHECK_RID_PREFIX)
        ]
        oldest = max(
            (time.perf_counter() - s.time_stats.created_time for s in states),
            default=0.0,
        )
        return (
            f"In-flight requests: {len(states)}, oldest {oldest:.0f}s; "
            f"last scheduler message {time.time() - tm.last_receive_tstamp:.0f}s ago."
        )

    def _run_bounded(self, name: str, fn: Callable[[], object]) -> None:
        """Run a diagnostic step so a hang or error in it cannot block the exit."""

        def target():
            try:
                fn()
            except BaseException:
                logger.exception(f"Frontend stall watchdog: {name} failed")

        thread = threading.Thread(target=target, daemon=True, name=f"stall-{name}")
        thread.start()
        thread.join(self._diagnostics_timeout)
        if thread.is_alive():
            logger.error(
                f"Frontend stall watchdog: {name} did not finish within "
                f"{self._diagnostics_timeout}s; continuing to exit."
            )


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass
    sys.stdout.flush()
    sys.stderr.flush()
