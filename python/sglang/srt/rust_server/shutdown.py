"""Main-process side of the Rust server's SIGTERM drain.

In Python mode the tokenizer manager drains on SIGTERM
(``TokenizerManager.sigterm_watchdog``): health answers 503, requests are
still admitted, every admitted request finishes, then a ``ShutdownReq`` stops
the schedulers, which get ``_SCHEDULER_EXIT_TIMEOUT_SECS`` to exit before the
process tree is killed and the server exits 0.

In Rust mode the listener lives in the scheduler process that hosts it, so the
main process only relays: it forwards SIGTERM to every local listener-hosting
scheduler, which drains its listener (``RustServer.install_sigterm_drain``)
and, on global rank 0, broadcasts the ``ShutdownReq`` once the listener has
closed. The main process then waits for the schedulers with Python's bound,
counted from the first scheduler exit or the rank-0 listener reporting that
its drain ended, whichever comes first. The report is a datagram the Rust
runtime sends as its listeners close, the moment before the ``ShutdownReq``
goes out; it also arrives from a wedged scheduler, whose listener closes
without waiting when its last health probe failed, as Python's drain skips.
As in Python, a scheduler still releasing its resources when the bound
expires is killed mid-release; with several nodes, the other nodes' ranks can
then hang in a collective until the orchestrator kills them.

With attention DP, the launcher of every node relays SIGTERM to the listeners
on its node. The MLP-sync gather of every step carries each DP listener's
drain state (``DPDrainColumn``), so a drain begun on any listener begins on
all of them, and rank 0 broadcasts the ``ShutdownReq`` only once every DP
listener has closed (``RustServer.on_drain_rows``). Rust mode also forces
``enable_dp_attention_local_control_broadcast`` on, which keeps the
``ShutdownReq`` in DP group 0; the MLP-sync gather of that step carries it to
the other DP groups (``DPShutdownVote``), so every rank on every node leaves
the event loop at the same step.

A launcher that got no SIGTERM (another node's, or one scheduler's own, began
the drain) treats its first scheduler exiting 0 as that gang shutdown: it
stops its subprocess watchdog and bounds the other schedulers by the same
exit bound. A crashed scheduler exits non-zero after sending SIGQUIT to this
process, whose handler kills the process tree; with a
``--custom-sigquit-handler`` that does not, the same bound applies with the
watchdog left armed, so the other schedulers are killed once it expires and
the launcher returns. Without a local
listener-hosting scheduler (elastic EP joiners and plain data parallelism,
whose schedulers the DP controller starts) the main process keeps its previous
SIGTERM behavior.
"""

from __future__ import annotations

import logging
import multiprocessing.connection
import os
import signal
import socket
import time
from typing import Any, List, Optional

from sglang.srt.managers.tokenizer_manager import _SCHEDULER_EXIT_TIMEOUT_SECS

logger = logging.getLogger(__name__)

# How often the wait re-checks for a SIGTERM.
_POLL_SECS = 1.0
# How long the killed schedulers get to be reaped: Python's drain reaps them
# through ``kill_process_tree``, whose default ``wait_timeout`` this is.
_REAP_TIMEOUT_SECS = 60


class RustServerSigtermDrain:
    """Relays SIGTERM to the Rust listeners and waits for the schedulers."""

    def __init__(
        self,
        scheduler_procs: List[Any],
        listener_procs: List[Any],
        subprocess_watchdog: Optional[Any],
        notify_name: str,
    ):
        self._scheduler_procs = scheduler_procs
        self._listener_procs = listener_procs
        self._subprocess_watchdog = subprocess_watchdog
        self._sigterm_received = False
        self._drain_ended = False
        self._notify = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._notify.bind("\0" + notify_name)
        self._notify.setblocking(False)

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self._on_sigterm)

    def _on_sigterm(self, signum=None, frame=None) -> None:
        if self._sigterm_received:
            return
        self._sigterm_received = True
        self._stop_watchdog()
        for proc in self._listener_procs:
            if proc.exitcode is None:
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def _stop_watchdog(self) -> None:
        # Child exits are expected from now on, not crashes.
        if self._subprocess_watchdog is not None:
            self._subprocess_watchdog.stop()

    def _poll_notify(self) -> None:
        # The abstract socket has no permissions: any local process can send
        # to it, so a datagram counts only once this process got SIGTERM.
        try:
            while self._notify.recv(64):
                if self._sigterm_received:
                    self._drain_ended = True
        except BlockingIOError:
            pass

    def wait(self) -> None:
        """Block until every scheduler exited, or, once draining (or once a
        scheduler exited without a SIGTERM here), until Python's
        scheduler-exit bound expires; the schedulers still alive then
        are killed and reaped, as Python's drain does, so a caller without the
        CLI's process-tree cleanup (a multiprocessing target) does not wait on
        them at exit. As in Python, a scheduler not reaped within
        ``_REAP_TIMEOUT_SECS`` raises ``RuntimeError``."""
        procs = self._scheduler_procs
        exit_deadline = None
        announced = False
        while alive := [proc for proc in procs if proc.is_alive()]:
            self._poll_notify()
            if self._sigterm_received and not announced:
                announced = True
                logger.warning(
                    "SIGTERM received. Draining requests and shutting down... "
                    "(forwarded to the Rust listeners)"
                )
            if (
                exit_deadline is None
                and self._sigterm_received
                and (len(alive) < len(procs) or self._drain_ended)
            ):
                exit_deadline = time.monotonic() + _SCHEDULER_EXIT_TIMEOUT_SECS
            elif (
                exit_deadline is None
                and not self._sigterm_received
                and any(proc.exitcode == 0 for proc in procs)
            ):
                # Killing the stragglers must not read as a crash.
                self._stop_watchdog()
                exit_deadline = time.monotonic() + _SCHEDULER_EXIT_TIMEOUT_SECS
                logger.warning(
                    "Scheduler exited without SIGTERM; treating it as a gang "
                    "shutdown, bounding the others by "
                    f"{_SCHEDULER_EXIT_TIMEOUT_SECS}s"
                )
            elif (
                exit_deadline is None
                and not self._sigterm_received
                and any(proc.exitcode is not None for proc in procs)
            ):
                # A crash: the watchdog stays armed so it is still reported.
                exit_deadline = time.monotonic() + _SCHEDULER_EXIT_TIMEOUT_SECS
                logger.warning(
                    "Scheduler crashed; bounding the others by "
                    f"{_SCHEDULER_EXIT_TIMEOUT_SECS}s"
                )
            if exit_deadline is not None and time.monotonic() >= exit_deadline:
                logger.warning(
                    f"Schedulers still alive {_SCHEDULER_EXIT_TIMEOUT_SECS}s "
                    "after the drain, killing them: "
                    f"{[proc.pid for proc in alive]}"
                )
                for proc in alive:
                    proc.kill()
                reap_deadline = time.monotonic() + _REAP_TIMEOUT_SECS
                for proc in alive:
                    proc.join(max(0.0, reap_deadline - time.monotonic()))
                if unreaped := [proc.pid for proc in alive if proc.is_alive()]:
                    raise RuntimeError(
                        f"{len(unreaped)} scheduler(s) not reaped within "
                        f"{_REAP_TIMEOUT_SECS}s after SIGKILL; pids={unreaped}"
                    )
                return
            multiprocessing.connection.wait(
                [proc.sentinel for proc in alive] + [self._notify],
                timeout=_POLL_SECS,
            )
        log = logger.info if self._sigterm_received else logger.error
        for proc in procs:
            log(f"Scheduler {proc.pid} terminated with {proc.exitcode}")
