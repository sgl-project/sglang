"""Tests for the macOS parent-death watchdog."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from sglang.srt.hardware_backend.mlx import parent_watchdog
from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

# Loads the watchdog straight from its source file: it only needs os/select/
# signal/threading, so the child skips the ~5 s `import sglang` and the orphan
# window stays short enough to keep the test cheap.
_CHILD_SCRIPT = """
import importlib.util, os, sys, time

module_path, pid_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("parent_watchdog", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if os.fork() == 0:
    tmp = pid_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(os.getpid()))
    os.replace(tmp, pid_path)
    # Install only once actually orphaned, so the case under test is reached by
    # a precondition rather than by racing the parent's exit.
    while os.getppid() != 1:
        time.sleep(0.01)
    module.start_parent_death_watcher()
    time.sleep(60)
    os._exit(0)
os._exit(0)
"""


@unittest.skipUnless(sys.platform == "darwin", "kqueue watchdog is macOS-only")
class TestParentDeathWatcher(CustomTestCase):
    def test_already_orphaned_self_kills(self):
        """An orphan must die even if it installs the watchdog after losing its parent.

        macOS reparents an orphan to launchd as soon as the parent exits, so a
        watchdog installed after that point sees ppid 1. Registering NOTE_EXIT on
        PID 1 succeeds and then blocks forever, so the worker used to survive as
        an orphan holding unified memory and its port.
        """
        with (
            mock.patch.object(parent_watchdog.os, "getppid", return_value=1),
            mock.patch.object(parent_watchdog.os, "kill") as mock_kill,
            mock.patch.object(parent_watchdog.threading, "Thread") as mock_thread,
        ):
            parent_watchdog.start_parent_death_watcher()

        mock_kill.assert_called_once_with(os.getpid(), signal.SIGKILL)
        # No thread, or it would be parked on a NOTE_EXIT for launchd that never fires.
        mock_thread.assert_not_called()

    def test_live_parent_starts_watcher_without_killing(self):
        """A worker with a living parent must be watched, not killed."""
        with (
            mock.patch.object(parent_watchdog.os, "getppid", return_value=4242),
            mock.patch.object(parent_watchdog.os, "kill") as mock_kill,
            mock.patch.object(parent_watchdog.threading, "Thread") as mock_thread,
        ):
            parent_watchdog.start_parent_death_watcher()

        mock_kill.assert_not_called()
        mock_thread.assert_called_once()
        self.assertTrue(mock_thread.call_args.kwargs["daemon"])
        mock_thread.return_value.start.assert_called_once()

    def test_orphaned_child_process_is_reaped(self):
        """End-to-end: a process orphaned before installing the watchdog must not leak."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pid_path = os.path.join(tmpdir, "child.pid")
            intermediate = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _CHILD_SCRIPT,
                    parent_watchdog.__file__,
                    pid_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            intermediate.wait(timeout=60)

            child_pid = self._read_pid(pid_path)
            self.addCleanup(self._force_kill, child_pid)
            self.assertTrue(
                self._wait_until_gone(child_pid, timeout=30),
                msg=f"orphaned pid {child_pid} survived; the watchdog did not fire",
            )

    def _read_pid(self, pid_path, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return int(open(pid_path).read())
            except (FileNotFoundError, ValueError):
                time.sleep(0.05)
        self.fail(f"child never reported its pid to {pid_path}")

    @staticmethod
    def _wait_until_gone(pid, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.05)
        return False

    @staticmethod
    def _force_kill(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    unittest.main()
