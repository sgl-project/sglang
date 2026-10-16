# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import psutil

PROCESS_PATH = Path(__file__).parents[2] / "runtime" / "utils" / "process.py"
SPEC = importlib.util.spec_from_file_location("multimodal_process", PROCESS_PATH)
assert SPEC is not None and SPEC.loader is not None
PROCESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESS)
kill_process_tree = PROCESS.kill_process_tree


@unittest.skipUnless(os.name == "posix", "requires POSIX process semantics")
class TestKillProcessTree(unittest.TestCase):
    def setUp(self):
        self.children = []

    def tearDown(self):
        for process in self.children:
            # let the tree root reap its children before terminating the root
            kill_process_tree(process.pid, include_parent=False)
            kill_process_tree(process.pid)
            process.wait(timeout=10)

    def spawn_child(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"]
        )
        self.children.append(process)
        return process.pid

    def spawn_tree(self):
        script = textwrap.dedent(
            """
            import subprocess
            import sys
            import time

            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(600)"]
            )
            print(child.pid, flush=True)
            child.wait()
            time.sleep(600)
            """
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
        )
        self.children.append(process)
        with process.stdout:
            child_pid = int(process.stdout.readline())
        return process.pid, child_pid

    def test_parent_exits_during_child_enumeration(self):
        parent_pid = 12345
        parent = mock.Mock()
        parent.children.side_effect = psutil.NoSuchProcess(parent_pid)
        with mock.patch.object(PROCESS.psutil, "Process", return_value=parent):
            kill_process_tree(parent_pid)
        parent.children.assert_called_once_with(recursive=True)
        parent.kill.assert_not_called()

    def test_waits_for_killed_children(self):
        parent_pid, child_pid = self.spawn_tree()
        sibling_pid = self.spawn_child()

        kill_process_tree(parent_pid, include_parent=False)

        self.assertFalse(psutil.pid_exists(child_pid))
        self.assertTrue(psutil.pid_exists(parent_pid))
        self.assertTrue(psutil.pid_exists(sibling_pid))

    def test_skip_pid_keeps_child_alive(self):
        parent_pid, child_pid = self.spawn_tree()
        sibling_pid = self.spawn_child()

        kill_process_tree(parent_pid, include_parent=False, skip_pid=child_pid)

        self.assertTrue(psutil.pid_exists(child_pid))
        self.assertTrue(psutil.pid_exists(parent_pid))
        self.assertTrue(psutil.pid_exists(sibling_pid))

    def test_waits_for_included_parent(self):
        child_pid = self.spawn_child()

        kill_process_tree(child_pid)

        self.assertFalse(psutil.pid_exists(child_pid))

    def test_missing_parent_is_noop(self):
        kill_process_tree(0x7FFFFFFF)


if __name__ == "__main__":
    unittest.main()
