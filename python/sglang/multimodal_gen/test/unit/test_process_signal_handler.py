# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import signal
import unittest
from pathlib import Path
from unittest import mock

PROCESS_PATH = Path(__file__).parents[2] / "runtime" / "utils" / "process.py"
SPEC = importlib.util.spec_from_file_location("multimodal_process", PROCESS_PATH)
assert SPEC is not None and SPEC.loader is not None
PROCESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESS)


@unittest.skipUnless(hasattr(os, "fork"), "requires POSIX process semantics")
class TestKillProcessTreeSignalHandler(unittest.TestCase):
    def setUp(self):
        self.previous = signal.getsignal(signal.SIGCHLD)

        def handler(signum, frame):
            pass

        self.handler = handler
        signal.signal(signal.SIGCHLD, handler)

    def tearDown(self):
        signal.signal(signal.SIGCHLD, self.previous)

    def test_missing_parent_preserves_handler(self):
        PROCESS.kill_process_tree(0x7FFFFFFF)
        self.assertIs(signal.getsignal(signal.SIGCHLD), self.handler)

    def test_existing_child_preserves_handler(self):
        child_pid = os.fork()
        if child_pid == 0:
            signal.pause()
            os._exit(0)

        try:
            PROCESS.kill_process_tree(child_pid)
            self.assertIs(signal.getsignal(signal.SIGCHLD), self.handler)
        finally:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child_pid, 0)
            except ChildProcessError:
                # psutil.wait_procs may already have reaped the child.
                pass

    def test_wait_timeout_preserves_handler(self):
        parent = mock.Mock(pid=12345)
        parent.children.return_value = []
        with (
            mock.patch.object(PROCESS.psutil, "Process", return_value=parent),
            mock.patch.object(
                PROCESS.psutil, "wait_procs", return_value=([], [parent])
            ) as wait,
        ):
            with self.assertRaisesRegex(RuntimeError, "did not exit"):
                PROCESS.kill_process_tree(parent.pid, wait_timeout=0.01)
        wait.assert_called_once_with([parent], timeout=0.01)
        self.assertIs(signal.getsignal(signal.SIGCHLD), self.handler)


if __name__ == "__main__":
    unittest.main()
