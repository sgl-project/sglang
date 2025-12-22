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
    def __init__(
        self,
        debug_name: str,
        get_counter: Callable[[], int],
        is_active: Callable[[], bool],
        watchdog_timeout: float,
        soft: bool = False,
        dump_info: Optional[Callable[[], str]] = None,
    ):
        self.debug_name = debug_name
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
            time.sleep(self.watchdog_timeout / 2)

        if self.dump_info is not None and (info_msg := self.dump_info()):
            logger.error(f"{self.debug_name} debug info:\n{info_msg}")

        pyspy_dump_schedulers()
        logger.error(
            f"{self.debug_name} watchdog timeout "
            f"({self.watchdog_timeout=}, {self.soft=})"
        )
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        if not self.soft:
            # Wait for some time so that the parent process can print the error.
            time.sleep(5)
            self.parent_process.send_signal(signal.SIGQUIT)
