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
"""Tests for subprocess_monitor.py"""

import multiprocessing as mp
import os
import threading
import time
import unittest

from sglang.srt.utils.subprocess_monitor import SubprocessMonitor, create_subprocess_monitor


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


class TestSubprocessMonitor(unittest.TestCase):
    def test_healthy_processes_no_callback(self):
        """Test that healthy processes don't trigger the crash callback."""
        # Start a healthy worker
        proc = mp.Process(target=healthy_worker)
        proc.start()

        callback_triggered = threading.Event()

        def on_crash():
            callback_triggered.set()

        monitor = SubprocessMonitor(
            processes=[proc],
            on_crash=on_crash,
            interval=0.1,
            process_names=["test_worker"],
        )
        monitor.start()

        # Wait a bit - callback should NOT be triggered
        time.sleep(0.5)
        self.assertFalse(callback_triggered.is_set())

        # Cleanup
        monitor.stop()
        proc.terminate()
        proc.join(timeout=1)

    def test_crashed_process_triggers_callback(self):
        """Test that a crashed process triggers the crash callback."""
        # Start a worker that will crash after a delay
        proc = mp.Process(target=slow_crash_worker, args=(0.2,))
        proc.start()

        callback_triggered = threading.Event()

        def on_crash():
            callback_triggered.set()

        monitor = SubprocessMonitor(
            processes=[proc],
            on_crash=on_crash,
            interval=0.1,
            process_names=["crashing_worker"],
        )
        monitor.start()

        # Wait for the callback to be triggered
        self.assertTrue(
            callback_triggered.wait(timeout=2.0),
            "Crash callback was not triggered within timeout",
        )

        # Cleanup
        monitor.stop()

    def test_immediate_crash_detection(self):
        """Test that an immediately crashing process is detected."""
        proc = mp.Process(target=crashing_worker)
        proc.start()

        callback_triggered = threading.Event()

        def on_crash():
            callback_triggered.set()

        monitor = SubprocessMonitor(
            processes=[proc],
            on_crash=on_crash,
            interval=0.05,
            process_names=["immediate_crash"],
        )
        monitor.start()

        # Should detect crash quickly
        self.assertTrue(
            callback_triggered.wait(timeout=1.0),
            "Immediate crash was not detected",
        )
        monitor.stop()

    def test_multiple_processes_one_crashes(self):
        """Test monitoring multiple processes where one crashes."""
        # One healthy, one crashing
        healthy_proc = mp.Process(target=healthy_worker)
        crashing_proc = mp.Process(target=slow_crash_worker, args=(0.2,))

        healthy_proc.start()
        crashing_proc.start()

        callback_triggered = threading.Event()

        def on_crash():
            callback_triggered.set()

        monitor = SubprocessMonitor(
            processes=[healthy_proc, crashing_proc],
            on_crash=on_crash,
            interval=0.1,
            process_names=["healthy", "crashing"],
        )
        monitor.start()

        # Should detect the crash
        self.assertTrue(
            callback_triggered.wait(timeout=2.0),
            "Crash was not detected when one of multiple processes crashed",
        )

        # Cleanup
        monitor.stop()
        healthy_proc.terminate()
        healthy_proc.join(timeout=1)

    def test_create_subprocess_monitor_factory(self):
        """Test the factory function for creating monitors."""
        proc1 = mp.Process(target=healthy_worker)
        proc2 = mp.Process(target=healthy_worker)
        detoken_proc = mp.Process(target=healthy_worker)

        proc1.start()
        proc2.start()
        detoken_proc.start()

        def on_crash():
            pass

        monitor = create_subprocess_monitor(
            scheduler_procs=[proc1, proc2],
            detokenizer_proc=detoken_proc,
            on_crash=on_crash,
        )

        # Check the monitor was configured correctly
        self.assertEqual(len(monitor._processes), 3)
        self.assertEqual(
            monitor._process_names, ["scheduler_0", "scheduler_1", "detokenizer"]
        )

        # Cleanup
        proc1.terminate()
        proc2.terminate()
        detoken_proc.terminate()
        proc1.join(timeout=1)
        proc2.join(timeout=1)
        detoken_proc.join(timeout=1)

    def test_empty_processes_list(self):
        """Test that monitor handles empty process list gracefully."""
        callback_triggered = threading.Event()

        def on_crash():
            callback_triggered.set()

        monitor = SubprocessMonitor(
            processes=[],
            on_crash=on_crash,
            interval=0.1,
        )
        monitor.start()

        # Should not crash and should not trigger callback
        time.sleep(0.3)
        self.assertFalse(callback_triggered.is_set())
        monitor.stop()


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method("spawn", force=True)
    unittest.main()
