from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from multiprocessing import Process
from typing import Callable, List, Optional

import psutil

from sglang.srt.utils.cudacore_pyspy_dump_utils import pyspy_dump_schedulers

logger = logging.getLogger(__name__)


class Watchdog:
    @staticmethod
    def create(
        debug_name: str,
        watchdog_timeout: Optional[float],
        soft: bool = False,
        test_stuck_time: float = 0,
    ) -> Watchdog:
        if watchdog_timeout is None:
            assert test_stuck_time == 0, (
                f"stuck tester can be enabled only if soft watchdog is enabled."
            )
            return _WatchdogNoop()
        return _WatchdogReal(
            debug_name=debug_name,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
            test_stuck_time=test_stuck_time,
        )

    def feed(self):
        pass

    @contextmanager
    def disable(self):
        yield


class _WatchdogReal(Watchdog):
    def __init__(
        self,
        debug_name: str,
        watchdog_timeout: float,
        soft: bool = False,
        test_stuck_time: float = 0,
    ):
        self._counter = 0
        self._active = True
        self._test_stuck_time = test_stuck_time
        self._test_stuck_triggered = False
        self._raw = WatchdogRaw(
            debug_name=debug_name,
            get_counter=lambda: self._counter,
            is_active=lambda: self._active,
            watchdog_timeout=watchdog_timeout,
            soft=soft,
        )
        logger.info(f"Watchdog {self._raw.debug_name} initialized.")
        if self._test_stuck_time > 0:
            logger.info(
                f"Watchdog {self._raw.debug_name} is configured to use {test_stuck_time=}."
            )

    def feed(self):
        # Only trigger the test stuck behavior once to avoid blocking server
        # startup health checks while still testing watchdog timeout detection
        if self._test_stuck_time > 0 and not self._test_stuck_triggered:
            self._test_stuck_triggered = True
            logger.info(
                f"Watchdog {self._raw.debug_name} start deliberately stuck for {self._test_stuck_time}s"
            )
            time.sleep(self._test_stuck_time)
            logger.info(
                f"Watchdog {self._raw.debug_name} end deliberately stuck for {self._test_stuck_time}s"
            )

        self._counter += 1

    @contextmanager
    def disable(self):
        assert self._active
        self._active = False
        try:
            yield
        finally:
            assert not self._active
            self._active = True


class _WatchdogNoop(Watchdog):
    pass


class WatchdogRaw:
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
        try:
            while True:
                self._watchdog_once()
        except Exception as e:
            logger.error(
                f"{self.debug_name} watchdog thread crashed: {e}", exc_info=True
            )

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


class SubprocessWatchdog:
    """Monitors subprocess liveness and triggers SIGQUIT when a crash is detected.

    When a subprocess crashes (e.g., NCCL timeout causing C++ std::terminate()),
    Python exception handlers never run, leaving the main process as a zombie
    service. This watchdog polls subprocess liveness in a daemon thread and
    sends SIGQUIT to trigger proper cleanup.

    See: https://github.com/sgl-project/sglang/issues/18421

    An optional ``on_exit`` callback is invoked before the default SIGQUIT path.
    Callers that can isolate one failed subprocess may disable that fail-stop
    while retaining polling for the remaining subprocesses.
    """

    def __init__(
        self,
        processes: List[Process],
        process_names: Optional[List[str]] = None,
        on_exit: Optional[Callable[[int, Process, str], None]] = None,
        fail_stop_on_exit: bool = True,
        interval: float = 1.0,
        on_poll: Optional[Callable[[], None]] = None,
        on_thread_stop: Optional[Callable[[], None]] = None,
        report_clean_exit: bool = False,
    ):
        self._processes = processes
        self._names = process_names or [f"process_{i}" for i in range(len(processes))]
        self._interval = interval
        self._on_exit = on_exit
        self._fail_stop_on_exit = fail_stop_on_exit
        self._on_poll = on_poll
        self._on_thread_stop = on_thread_stop
        self._report_clean_exit = report_clean_exit
        self._stop_event = threading.Event()
        self._reported = set()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None or not self._processes:
            return
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="subprocess-watchdog"
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=self._interval * 2)
            self._thread = None

    def _on_thread_start(self) -> None:
        pass

    def _monitor_loop(self) -> None:
        try:
            self._on_thread_start()
            while not self._stop_event.wait(self._interval):
                if self._check_processes():
                    return
                if self._on_poll is None and len(self._reported) == len(
                    self._processes
                ):
                    return
                if self._on_poll is not None:
                    self._on_poll()
        except Exception as e:
            logger.error(f"SubprocessWatchdog thread crashed: {e}", exc_info=True)
        finally:
            if self._on_thread_stop is not None:
                self._on_thread_stop()

    def _check_processes(self) -> bool:
        for index, (proc, name) in enumerate(zip(self._processes, self._names)):
            if index in self._reported or proc.is_alive() or proc.exitcode is None:
                continue
            self._reported.add(index)
            if proc.exitcode == 0 and not self._report_clean_exit:
                continue
            if self._handle_process_exit(index, proc, name):
                return True
        return False

    def _handle_process_exit(self, index: int, proc: Process, name: str) -> bool:
        if self._on_exit is not None:
            self._on_exit(index, proc, name)

        if proc.exitcode == 0:
            return False

        if not self._fail_stop_on_exit:
            logger.warning(f"Subprocess {name} (pid={proc.pid}) crashed with exit code {proc.exitcode}.")
            return False

        logger.error(
            f"Subprocess {name} (pid={proc.pid}) crashed "
            f"with exit code {proc.exitcode}. "
            f"Triggering SIGQUIT for cleanup..."
        )
        os.kill(os.getpid(), signal.SIGQUIT)
        return True
