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
"""Tests for SubprocessWatchdog in watchdog.py"""

import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import unittest.mock

from sglang.srt.utils.watchdog import SubprocessWatchdog, wait_for_subprocess_startup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=24, suite="base-a-test-cpu")
register_cpu_ci(est_time=18, suite="stage-b-test-cpu-intel")


def healthy_worker():
    time.sleep(10)


def crashing_worker():
    os._exit(1)


def slow_crash_worker(delay: float = 0.5):
    time.sleep(delay)
    os._exit(42)


def noop_worker():
    pass


class TestSubprocessWatchdog(CustomTestCase):
    def setUp(self):
        self.sigquit_triggered = threading.Event()
        self._procs = []
        self._monitor = None

        original_kill = os.kill

        def mock_kill(pid, sig):
            if sig == signal.SIGQUIT:
                self.sigquit_triggered.set()
            else:
                original_kill(pid, sig)

        self._patcher = unittest.mock.patch("os.kill", side_effect=mock_kill)
        self._patcher.start()

    def tearDown(self):
        if self._monitor is not None:
            self._monitor.stop()
        self._patcher.stop()
        for p in self._procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

    def _spawn(self, target, args=()):
        proc = mp.Process(target=target, args=args)
        proc.start()
        self._procs.append(proc)
        return proc

    def _watch(self, procs, names=None, interval=0.1):
        if not isinstance(procs, list):
            procs = [procs]
        self._monitor = SubprocessWatchdog(
            processes=procs,
            process_names=names,
            interval=interval,
        )
        self._monitor.start()
        return self._monitor

    def test_healthy_processes_no_sigquit(self):
        proc = self._spawn(healthy_worker)
        self._watch(proc)
        time.sleep(0.5)
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_crashed_process_triggers_sigquit(self):
        proc = self._spawn(slow_crash_worker, args=(0.2,))
        self._watch(proc)
        self.assertTrue(
            self.sigquit_triggered.wait(timeout=5.0),
            "SIGQUIT was not triggered within timeout",
        )

    def test_immediate_crash_detection(self):
        proc = self._spawn(crashing_worker)
        self._watch(proc, interval=0.05)
        self.assertTrue(
            self.sigquit_triggered.wait(timeout=5.0),
            "Immediate crash was not detected",
        )

    def test_multiple_processes_one_crashes(self):
        healthy = self._spawn(healthy_worker)
        crashing = self._spawn(slow_crash_worker, args=(0.2,))
        self._watch([healthy, crashing], names=["healthy", "crashing"])
        self.assertTrue(
            self.sigquit_triggered.wait(timeout=5.0),
            "Crash was not detected when one of multiple processes crashed",
        )

    def test_empty_processes_list(self):
        self._watch([], interval=0.1)
        time.sleep(0.3)
        self.assertFalse(self.sigquit_triggered.is_set())

    def test_normal_exit_no_sigquit(self):
        proc = self._spawn(noop_worker)
        proc.join(timeout=2)
        self._watch(proc)
        time.sleep(0.3)
        self.assertFalse(
            self.sigquit_triggered.is_set(),
            "SIGQUIT should not be triggered for normal exit (exitcode=0)",
        )

    def test_clean_exit_is_fatal_only_during_startup(self):
        proc = self._spawn(noop_worker)
        proc.join(timeout=10)
        self.assertEqual(proc.exitcode, 0)
        for ready in (False, True):
            with self.subTest(ready=ready):
                self.sigquit_triggered.clear()
                monitor = SubprocessWatchdog([proc], startup=True)
                if ready:
                    monitor.mark_startup_complete()
                self.assertEqual(monitor._check_processes(), not ready)
                self.assertEqual(self.sigquit_triggered.is_set(), not ready)


class TestSubprocessStartup(CustomTestCase):
    def test_later_rank_exit_before_ready(self):
        """Even exit code zero is a startup failure if a rank never sends ready."""
        context = mp.get_context("spawn")
        for exitcode in (0, 23):
            with self.subTest(exitcode=exitcode):
                release = context.Event()
                processes = [
                    context.Process(target=release.wait, args=(30,)),
                    context.Process(target=os._exit, args=(exitcode,)),
                ]
                pipes = [context.Pipe(duplex=False) for _ in processes]
                results = queue.Queue()

                def wait():
                    try:
                        results.put(
                            wait_for_subprocess_startup(
                                [reader for reader, _ in pipes], processes
                            )
                        )
                    except Exception as error:
                        results.put(error)

                waiter = threading.Thread(target=wait, daemon=True)
                try:
                    for process in processes:
                        process.start()
                    processes[1].join(timeout=10)
                    self.assertEqual(processes[1].exitcode, exitcode)
                    # Both writers stay open: neither EOF nor rank 0 can unblock recv.
                    waiter.start()
                    error = results.get(timeout=10)
                    self.assertIsInstance(error, RuntimeError)
                    self.assertIn("Rank 1", str(error))
                    self.assertIn(f"exit code: {exitcode}", str(error))
                    self.assertTrue(processes[0].is_alive())
                finally:
                    for _, writer in pipes:
                        writer.send({"status": "ready"})
                    if waiter.ident is not None:
                        waiter.join(timeout=2)
                    release.set()
                    for process in processes:
                        if process.pid is not None:
                            process.join(timeout=2)
                            if process.is_alive():
                                process.kill()
                                process.join(timeout=2)
                    for reader, writer in pipes:
                        reader.close()
                        writer.close()
                self.assertFalse(waiter.is_alive())


if __name__ == "__main__":
    import unittest

    unittest.main()
