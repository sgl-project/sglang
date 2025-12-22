"""Generic watchdog utilities for monitoring process health."""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from typing import Callable, Optional

import psutil

from sglang.srt.utils.common import pyspy_dump_schedulers

logger = logging.getLogger(__name__)


class ProcessWatchdog:
    """A generic watchdog that monitors a counter to detect stuck processes.

    The watchdog runs in a background thread and checks if the process is making
    progress by monitoring a counter. If the counter doesn't change within the
    timeout period while the process is active, it triggers a warning (soft mode)
    or kills the process (hard mode).
    """

    def __init__(
        self,
        process_name: str,
        get_counter: Callable[[], int],
        is_active: Callable[[], bool],
        watchdog_timeout: float,
        soft: bool = False,
        dump_info: Optional[Callable[[], str]] = None,
    ):
        """Initialize the watchdog.

        Args:
            process_name: Name of the process for logging.
            get_counter: A callable that returns the current activity counter.
            is_active: A callable that returns True if the process should be active
                      (i.e., watchdog should check for progress).
            watchdog_timeout: Timeout in seconds before triggering.
            soft: If True, only log warnings. If False, kill the process.
            dump_info: Optional callable to dump additional debug info.
        """
        self.process_name = process_name
        self.get_counter = get_counter
        self.is_active = is_active
        self.watchdog_timeout = watchdog_timeout
        self.soft = soft
        self.dump_info = dump_info

        self.parent_process = psutil.Process().parent()
        t = threading.Thread(target=self._watchdog_thread, daemon=True)
        t.start()

    def _watchdog_thread(self):
        while True:
            self._watchdog_once()

    def _watchdog_once(self):
        watchdog_last_counter = 0
        watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.is_active():
                current_counter = self.get_counter()
                if watchdog_last_counter == current_counter:
                    if current > watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    watchdog_last_counter = current_counter
                    watchdog_last_time = current
            else:
                watchdog_last_time = current
            time.sleep(self.watchdog_timeout / 2)

        if self.dump_info is not None:
            try:
                info_msg = self.dump_info()
                logger.error(f"{self.process_name} debug info:\n{info_msg}")
            except Exception as e:
                logger.error(f"Failed to dump debug info: {e}")

        pyspy_dump_schedulers()
        logger.error(
            f"{self.process_name} watchdog timeout "
            f"({self.watchdog_timeout=}, {self.soft=})"
        )
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        if not self.soft:
            time.sleep(5)
            self.parent_process.send_signal(signal.SIGQUIT)

