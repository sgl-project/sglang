# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Subprocess liveness monitor for detecting scheduler/detokenizer crashes.

When a subprocess crashes (e.g., due to NCCL timeout triggering C++ std::terminate()),
Python exception handlers never run. This monitor polls subprocess liveness and triggers
the SIGQUIT handler when a crash is detected, ensuring proper cleanup of the zombie service.

See: https://github.com/sgl-project/sglang/issues/18421
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from multiprocessing import Process
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Default monitoring interval in seconds
DEFAULT_MONITOR_INTERVAL = 1.0


class SubprocessMonitor:
    """
    Monitors subprocess liveness and triggers cleanup when a crash is detected.

    This monitor runs in a background daemon thread and periodically checks if
    the monitored subprocesses are still alive. When a subprocess crash is detected,
    it triggers the provided on_crash callback (typically the SIGQUIT handler).
    """

    def __init__(
        self,
        processes: List[Process],
        on_crash: Callable[[], None],
        interval: float = DEFAULT_MONITOR_INTERVAL,
        process_names: Optional[List[str]] = None,
    ):
        """
        Args:
            processes: List of subprocess.Process objects to monitor
            on_crash: Callback function to invoke when a subprocess crash is detected
            interval: Polling interval in seconds (default: 1.0)
            process_names: Optional list of names for logging (e.g., ["scheduler_0", "scheduler_1"])
        """
        self._processes = processes
        self._on_crash = on_crash
        self._interval = interval
        self._process_names = process_names or [
            f"process_{i}" for i in range(len(processes))
        ]
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._thread is not None:
            logger.warning("SubprocessMonitor already started")
            return

        if not self._processes:
            logger.debug("No processes to monitor, skipping SubprocessMonitor start")
            return

        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="subprocess-monitor",
        )
        self._thread.start()
        logger.info(
            f"SubprocessMonitor started, monitoring {len(self._processes)} process(es) "
            f"with {self._interval}s interval"
        )

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            self._thread = None

    def _monitor_loop(self) -> None:
        """Main monitoring loop that checks subprocess liveness."""
        try:
            while not self._stop_event.is_set():
                for proc, name in zip(self._processes, self._process_names):
                    if not proc.is_alive():
                        exit_code = proc.exitcode
                        logger.error(
                            f"Subprocess {name} (pid={proc.pid}) crashed with exit code {exit_code}. "
                            f"Triggering cleanup..."
                        )
                        self._on_crash()
                        # After triggering on_crash (which should kill the process tree),
                        # we don't expect to continue, but return just in case
                        return

                self._stop_event.wait(self._interval)
        except Exception as e:
            logger.error(f"SubprocessMonitor thread crashed: {e}", exc_info=True)


def create_subprocess_monitor(
    scheduler_procs: List[Process],
    detokenizer_proc: Optional[Process],
    on_crash: Callable[[], None],
    interval: float = DEFAULT_MONITOR_INTERVAL,
) -> SubprocessMonitor:
    """
    Factory function to create a SubprocessMonitor for SGLang processes.

    Args:
        scheduler_procs: List of scheduler subprocess handles
        detokenizer_proc: Optional detokenizer subprocess handle
        on_crash: Callback to invoke on subprocess crash (typically triggers SIGQUIT handler)
        interval: Polling interval in seconds

    Returns:
        Configured SubprocessMonitor instance (not yet started)
    """
    processes = list(scheduler_procs)
    names = [f"scheduler_{i}" for i in range(len(scheduler_procs))]

    if detokenizer_proc is not None:
        processes.append(detokenizer_proc)
        names.append("detokenizer")

    return SubprocessMonitor(
        processes=processes,
        on_crash=on_crash,
        interval=interval,
        process_names=names,
    )
