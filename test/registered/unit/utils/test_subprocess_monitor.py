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
import unittest
import unittest.mock

from sglang.srt.utils.watchdog import SubprocessWatchdog


def healthy_worker():
    """A worker that stays alive for a while."""
    time.sleep(10)


def crashing_worker():
    """A worker that crashes immediately."""
    os._exit(1)


def slow_crash_worker(delay: float = 0.5):
    """A worker that crashes after a delay."""
    time.sleep(delay)
    os._exit(42)


class TestSubprocessWatchdog(unittest.TestCase):
    def test_healthy_processes_no_callback(self):
        """Test that healthy processes don't trigger SIGQUIT."""
        proc = mp.Process(target=healthy_worker)
        proc.start()

        callback_triggered = threading.Event()
        original_kill = os.kill

        def mock_kill(pid, sig):
            if sig == signal.SIGQUIT:
                callback_triggered.set()
            else:
                original_kill(pid, sig)

        monitor = SubprocessWatchdog(
            processes=[proc],
            process_names=["test_worker"],
            interval=0.1,
        )
        with unittest.mock.patch("os.kill", side_effect=mock_kill):
            monitor.start()
            time.sleep(0.5)

        self.assertFalse(callback_triggered.is_set())
        monitor.stop()
        proc.terminate()
        proc.join(timeout=1)

    def test_crashed_process_triggers_sigquit(self):
        """Test that a crashed process triggers SIGQUIT."""
        proc = mp.Process(target=slow_crash_worker, args=(0.2,))
        proc.start()

        callback_triggered = threading.Event()
        original_kill = os.kill

        def mock_kill(pid, sig):
            if sig == signal.SIGQUIT:
                callback_triggered.set()
            else:
                original_kill(pid, sig)

        monitor = SubprocessWatchdog(
            processes=[proc],
            process_names=["crashing_worker"],
            interval=0.1,
        )
        with unittest.mock.patch("os.kill", side_effect=mock_kill):
            monitor.start()
            self.assertTrue(
                callback_triggered.wait(timeout=2.0),
                "SIGQUIT was not triggered within timeout",
            )
        monitor.stop()

    def test_immediate_crash_detection(self):
        """Test that an immediately crashing process is detected."""
        proc = mp.Process(target=crashing_worker)
        proc.start()

        callback_triggered = threading.Event()
        original_kill = os.kill

        def mock_kill(pid, sig):
            if sig == signal.SIGQUIT:
                callback_triggered.set()
            else:
                original_kill(pid, sig)

        monitor = SubprocessWatchdog(
            processes=[proc],
            process_names=["immediate_crash"],
            interval=0.05,
        )
        with unittest.mock.patch("os.kill", side_effect=mock_kill):
            monitor.start()
            self.assertTrue(
                callback_triggered.wait(timeout=1.0),
                "Immediate crash was not detected",
            )
        monitor.stop()

    def test_multiple_processes_one_crashes(self):
        """Test monitoring multiple processes where one crashes."""
        healthy_proc = mp.Process(target=healthy_worker)
        crashing_proc = mp.Process(target=slow_crash_worker, args=(0.2,))

        healthy_proc.start()
        crashing_proc.start()

        callback_triggered = threading.Event()
        original_kill = os.kill

        def mock_kill(pid, sig):
            if sig == signal.SIGQUIT:
                callback_triggered.set()
            else:
                original_kill(pid, sig)

        monitor = SubprocessWatchdog(
            processes=[healthy_proc, crashing_proc],
            process_names=["healthy", "crashing"],
            interval=0.1,
        )
        with unittest.mock.patch("os.kill", side_effect=mock_kill):
            monitor.start()
            self.assertTrue(
                callback_triggered.wait(timeout=2.0),
                "Crash was not detected when one of multiple processes crashed",
            )
        monitor.stop()
        healthy_proc.terminate()
        healthy_proc.join(timeout=1)

    def test_empty_processes_list(self):
        """Test that watchdog handles empty process list gracefully."""
        callback_triggered = threading.Event()
        original_kill = os.kill

        def mock_kill(pid, sig):
            if sig == signal.SIGQUIT:
                callback_triggered.set()
            else:
                original_kill(pid, sig)

        monitor = SubprocessWatchdog(processes=[], interval=0.1)
        with unittest.mock.patch("os.kill", side_effect=mock_kill):
            monitor.start()
            time.sleep(0.3)

        self.assertFalse(callback_triggered.is_set())
        monitor.stop()


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method("spawn", force=True)
    unittest.main()
