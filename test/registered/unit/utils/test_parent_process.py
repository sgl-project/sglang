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
"""Tests for resolve_failure_notification_parent / notify_parent_of_failure
(sglang.srt.utils.process), the fix for #41539: a worker whose launcher dies
while the worker is still importing modules must not send SIGQUIT to PID 1.
"""

import json
import multiprocessing as mp
import os
import signal
import tempfile
import time
import unittest.mock

from sglang.srt.utils.process import (
    _multiprocessing_parent_alive,
    notify_parent_of_failure,
    resolve_failure_notification_parent,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=180, suite="base-a-test-cpu")


def _noop():
    pass


def _child_notify(result_path):
    """Spawned worker: notify the launcher and record the outcome."""
    notified = notify_parent_of_failure()
    parent = resolve_failure_notification_parent()
    with open(result_path, "w") as f:
        json.dump(
            {
                "notified": notified,
                "parent_pid": parent.pid if parent is not None else None,
                "ppid": os.getppid(),
            },
            f,
        )


def _child_notify_after_delay(result_path, delay):
    """Spawned worker: wait out the import window, then resolve the parent."""
    time.sleep(delay)
    parent = resolve_failure_notification_parent()
    notified = notify_parent_of_failure()
    with open(result_path, "w") as f:
        json.dump(
            {
                "notified": notified,
                "parent_pid": parent.pid if parent is not None else None,
                "ppid": os.getppid(),
            },
            f,
        )


def _launcher_spawns_worker_and_dies(ctx_name, result_path, worker_delay):
    """Intermediate launcher: spawn a worker, then exit before it resolves.

    Mirrors the #41539 trigger: the worker is still importing when the
    launcher exits, so the kernel reparents the worker to PID 1.
    """
    ctx = mp.get_context(ctx_name)
    worker = ctx.Process(
        target=_child_notify_after_delay, args=(result_path, worker_delay)
    )
    worker.start()
    time.sleep(0.3)
    os._exit(0)


class TestParentProcess(CustomTestCase):
    def test_live_launcher_is_notified(self):
        """A live launcher still receives the failure SIGQUIT."""
        for ctx_name in ("spawn", "fork"):
            with self.subTest(ctx_name=ctx_name):
                received = []
                old_handler = signal.getsignal(signal.SIGQUIT)

                def _handler(signum, frame):
                    received.append(signum)

                signal.signal(signal.SIGQUIT, _handler)
                try:
                    ctx = mp.get_context(ctx_name)
                    with tempfile.TemporaryDirectory() as tmpdir:
                        result_path = os.path.join(tmpdir, "result.json")
                        proc = ctx.Process(target=_child_notify, args=(result_path,))
                        proc.start()
                        proc.join(timeout=120)
                        self.assertEqual(proc.exitcode, 0)
                        with open(result_path) as f:
                            result = json.load(f)
                        self.assertTrue(result["notified"])
                        self.assertEqual(result["parent_pid"], os.getpid())
                        self.assertEqual(result["ppid"], os.getpid())
                        self.assertEqual(received, [signal.SIGQUIT])
                finally:
                    signal.signal(signal.SIGQUIT, old_handler)

    def test_dead_launcher_is_not_signalled(self):
        """An orphaned worker resolves no parent and signals nobody.

        The worker is reparented to PID 1 (the old code would SIGQUIT init);
        the fix must return None instead of the adoptive parent.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = os.path.join(tmpdir, "result.json")
            launcher = mp.Process(
                target=_launcher_spawns_worker_and_dies,
                args=("spawn", result_path, 2.0),
            )
            launcher.start()
            launcher.join(timeout=30)
            self.assertEqual(launcher.exitcode, 0)
            result = None
            deadline = time.time() + 120
            while time.time() < deadline:
                try:
                    with open(result_path) as f:
                        result = json.load(f)
                    break
                except (FileNotFoundError, json.JSONDecodeError):
                    time.sleep(0.2)
            self.assertIsNotNone(result, "worker never reported")
            self.assertFalse(result["notified"])
            self.assertIsNone(result["parent_pid"])
            self.assertEqual(result["ppid"], 1)

    def test_multiprocessing_parent_alive_sentinel(self):
        """The sentinel distinguishes a live launcher from a dead one."""
        read_fd, write_fd = os.pipe()
        try:
            fake_parent = unittest.mock.Mock()
            fake_parent.sentinel = read_fd
            # Write end open: parent alive.
            self.assertTrue(_multiprocessing_parent_alive(fake_parent))
            os.close(write_fd)
            write_fd = None
            # Write end closed: parent gone (sentinel readable).
            self.assertFalse(_multiprocessing_parent_alive(fake_parent))
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def test_multiprocessing_parent_alive_no_sentinel(self):
        """Without a sentinel, liveness falls back to PID existence."""
        proc = mp.Process(target=_noop)
        proc.start()
        proc.join()
        fake_parent = unittest.mock.Mock()
        fake_parent.sentinel = None
        fake_parent.pid = proc.pid
        self.assertFalse(_multiprocessing_parent_alive(fake_parent))
        fake_parent.pid = os.getpid()
        self.assertTrue(_multiprocessing_parent_alive(fake_parent))

    def test_fallback_outside_multiprocessing(self):
        """Outside multiprocessing, the OS parent is returned (as before)."""
        parent = resolve_failure_notification_parent()
        self.assertIsNotNone(parent)
        self.assertEqual(parent.pid, os.getppid())
