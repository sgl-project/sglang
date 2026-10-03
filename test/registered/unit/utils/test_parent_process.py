"""Regression tests for worker error notifications targeting the launcher."""

import multiprocessing
import os
import signal
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import psutil

from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _notify_parent(connection, launcher_pid):
    parent = common.get_parent_process()
    # Even with a regression, never signal a forkserver or an unrelated parent.
    if parent.pid == launcher_pid:
        parent.send_signal(signal.SIGUSR1)
    connection.send((parent.pid, os.getppid()))
    connection.close()


class TestGetParentProcess(CustomTestCase):
    def test_live_launcher_receives_notification(self):
        """Forkserver workers must notify the launcher, not their OS parent."""
        received_signals = []
        previous_handler = signal.signal(
            signal.SIGUSR1, lambda signum, frame: received_signals.append(signum)
        )
        try:
            for method in ("spawn", "forkserver"):
                if method not in multiprocessing.get_all_start_methods():
                    continue
                with self.subTest(start_method=method):
                    received_signals.clear()
                    ctx = multiprocessing.get_context(method)
                    reader, writer = ctx.Pipe(duplex=False)
                    child = ctx.Process(
                        target=_notify_parent, args=(writer, os.getpid())
                    )
                    try:
                        child.start()
                        writer.close()
                        self.assertTrue(reader.poll(60), "Worker did not report back")
                        parent_pid, os_parent_pid = reader.recv()
                        child.join(10)
                        self.assertEqual(child.exitcode, 0)
                        self.assertEqual(parent_pid, os.getpid())
                        self.assertEqual(received_signals, [signal.SIGUSR1])
                        if method == "forkserver":
                            self.assertNotEqual(os_parent_pid, os.getpid())
                    finally:
                        if child.is_alive():
                            child.kill()
                            child.join(10)
                        reader.close()
                        writer.close()
                        child.close()
        finally:
            signal.signal(signal.SIGUSR1, previous_handler)

    def test_missing_launcher_does_not_fall_back_to_adoptive_parent(self):
        """A launcher that exits before lookup must not redirect errors to init."""
        launcher_pid = 12345
        launcher = SimpleNamespace(pid=launcher_pid, is_alive=lambda: False)
        adoptive_parent = SimpleNamespace(pid=1)

        def process(pid=None):
            if pid == launcher_pid:
                raise psutil.NoSuchProcess(pid)
            return SimpleNamespace(parent=lambda: adoptive_parent)

        with (
            patch.object(common, "parent_process", return_value=launcher),
            patch.object(common.psutil, "Process", side_effect=process),
            self.assertRaises(psutil.NoSuchProcess) as raised,
        ):
            common.get_parent_process()
        self.assertEqual(raised.exception.pid, launcher_pid)

    def test_launcher_dies_while_pid_is_resolved(self):
        """A reused launcher PID must not become an unrelated signal target."""
        launcher_pid = 12345
        alive = True
        launcher = SimpleNamespace(pid=launcher_pid, is_alive=lambda: alive)

        def process(pid=None):
            nonlocal alive
            # The launcher exits and its PID is reused during psutil lookup.
            alive = False
            return SimpleNamespace(pid=pid, parent=lambda: SimpleNamespace(pid=1))

        with (
            patch.object(common, "parent_process", return_value=launcher),
            patch.object(common.psutil, "Process", side_effect=process),
            self.assertRaises(psutil.NoSuchProcess) as raised,
        ):
            common.get_parent_process()
        self.assertEqual(raised.exception.pid, launcher_pid)

    def test_pid_one_can_be_the_real_launcher(self):
        """A container launcher running as PID 1 still needs worker errors."""
        notifications = []
        launcher = SimpleNamespace(pid=1, is_alive=lambda: True)
        process = Mock(pid=1)
        process.send_signal.side_effect = lambda signum: notifications.append(
            (process.pid, signum)
        )
        current_process = SimpleNamespace(parent=lambda: process)
        with (
            patch.object(common, "parent_process", return_value=launcher),
            patch.object(
                common.psutil,
                "Process",
                side_effect=lambda pid=None: process if pid == 1 else current_process,
            ),
        ):
            common.get_parent_process().send_signal(signal.SIGQUIT)
        self.assertEqual(notifications, [(1, signal.SIGQUIT)])

    def test_non_multiprocessing_process_uses_os_parent(self):
        """Watchdogs outside multiprocessing retain their OS-parent target."""
        with patch.object(common, "parent_process", return_value=None):
            parent = common.get_parent_process()
        self.assertEqual(parent, psutil.Process().parent())


if __name__ == "__main__":
    unittest.main()
