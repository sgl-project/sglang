# SPDX-License-Identifier: Apache-2.0
"""Linux producer identity and early fail-stop protection for CUDA IPC."""

from __future__ import annotations

import errno
import logging
import math
import os
import select
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int  # /proc/<pid>/stat field 22, not wall-clock time

    @classmethod
    def read(cls, pid: int) -> ProcessIdentity:
        if type(pid) is not int or pid <= 0:
            raise ValueError("Producer PID must be a positive integer")
        # comm may itself contain spaces or ')'; fields following the final
        # ')' start at field 3. Zombies have lost their CUDA allocations.
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text.rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X", "x"):
            raise ProcessLookupError(f"Producer {pid} is a zombie/dead process")
        return cls(pid, int(fields[19]))

    def is_alive(self) -> bool:
        try:
            return self == self.read(self.pid)
        except (OSError, ValueError, IndexError):
            return False


class ProducerDiedError(RuntimeError):
    pass


class ProducerWatchdog:
    """Start BEFORE importing handles; close only after all mappings are unused.

    Producer death cannot be made race-free by a watcher. This provides bounded
    fail-stop detection, not permission to continue using a dead producer's
    allocations. A generation must also retain its tensors until consumers exit.
    """

    def __init__(
        self,
        identity: ProcessIdentity,
        *,
        poll_interval: float = 0.1,
        on_death: Callable[[], None] | None = None,
        use_pidfd: bool = True,
    ):
        if (
            type(poll_interval) not in (int, float)
            or not math.isfinite(poll_interval)
            or not 0 < poll_interval <= 10
        ):
            raise ValueError("Watchdog poll interval must be in (0, 10] seconds")
        self.identity = identity
        self._owner_pid = os.getpid()
        self._poll_interval = poll_interval
        self._on_death = on_death or self._kill_consumer
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._pidfd = None
        self._closed = False
        if not identity.is_alive():
            raise ProducerDiedError(f"Producer identity is no longer live: {identity}")
        if use_pidfd and hasattr(os, "pidfd_open"):
            try:
                self._pidfd = os.pidfd_open(identity.pid)
            except OSError as error:
                if error.errno not in (errno.ENOSYS, errno.EPERM, errno.EINVAL):
                    raise ProducerDiedError("Cannot open producer pidfd") from error
        if not identity.is_alive():
            if self._pidfd is not None:
                os.close(self._pidfd)
            raise ProducerDiedError("Producer changed while opening its pidfd")
        self._thread = threading.Thread(
            target=self._watch, name="weight-cache-producer-watchdog", daemon=True
        )
        try:
            self._thread.start()
        except Exception:
            if self._pidfd is not None:
                os.close(self._pidfd)
            raise

    @staticmethod
    def _kill_consumer() -> None:
        os.kill(os.getpid(), signal.SIGKILL)

    def _watch(self) -> None:
        try:
            while not self._stop.wait(self._poll_interval):
                if self._pidfd is not None:
                    dead = bool(select.select([self._pidfd], [], [], 0)[0])
                else:
                    dead = not self.identity.is_alive()
                if dead:
                    break
            else:
                return
        except Exception:
            logger.exception("Weight-cache producer monitoring failed")
        self._lost.set()
        logger.critical(
            "Weight-cache producer %s was lost; terminating its consumer",
            self.identity,
        )
        self._on_death()

    def check_alive(self) -> None:
        if (
            os.getpid() != self._owner_pid
            or self._closed
            or self._lost.is_set()
            or not self._thread.is_alive()
            or not self.identity.is_alive()
        ):
            raise ProducerDiedError(f"Producer guard is not live: {self.identity}")

    def close(self) -> None:
        """Only after the caller has stopped using every imported tensor."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._thread.join()
        if self._pidfd is not None:
            os.close(self._pidfd)
            self._pidfd = None
