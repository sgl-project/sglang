# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import time
import unittest
from pathlib import Path

import psutil

PROCESS_PATH = Path(__file__).parents[2] / "runtime" / "utils" / "process.py"
SPEC = importlib.util.spec_from_file_location("multimodal_process", PROCESS_PATH)
assert SPEC is not None and SPEC.loader is not None
PROCESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESS)
kill_process_tree = PROCESS.kill_process_tree


@unittest.skipUnless(hasattr(os, "fork"), "requires POSIX process semantics")
class TestKillProcessTree(unittest.TestCase):
    def setUp(self):
        self.children = []

    def tearDown(self):
        for pid in self.children:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass

    def spawn_child(self):
        pid = os.fork()
        if pid == 0:
            while True:
                time.sleep(60)
        self.children.append(pid)
        return pid

    def test_waits_for_killed_children(self):
        child_pid = self.spawn_child()

        kill_process_tree(os.getpid(), include_parent=False)

        self.assertFalse(psutil.pid_exists(child_pid))

    def test_skip_pid_keeps_child_alive(self):
        child_pid = self.spawn_child()

        kill_process_tree(os.getpid(), include_parent=False, skip_pid=child_pid)

        self.assertTrue(psutil.pid_exists(child_pid))

    def test_waits_for_included_parent(self):
        child_pid = self.spawn_child()

        kill_process_tree(child_pid)

        self.assertFalse(psutil.pid_exists(child_pid))

    def test_missing_parent_is_noop(self):
        kill_process_tree(0x7FFFFFFF)


if __name__ == "__main__":
    unittest.main()
