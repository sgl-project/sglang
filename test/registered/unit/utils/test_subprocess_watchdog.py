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
import signal
import threading
import time
import unittest.mock

from sglang.srt.utils.watchdog import SubprocessWatchdog
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


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


if __name__ == "__main__":
    import unittest

    unittest.main()
