import subprocess
import sys
import unittest

import psutil

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, terminate_and_kill_process_tree

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# A launched server is a launcher process owning worker children. The launcher
# can exit before the fallback kill runs, and a worker left behind holds GPU
# memory until the next test times out on its idle gate.
PARENT_SCRIPT = (
    "import subprocess, sys, time; "
    "child = subprocess.Popen("
    "[sys.executable, '-c', sys.argv[1]], "
    "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
    "print(child.pid, flush=True); "
    "time.sleep(60)"
)
STUBBORN_PARENT_SCRIPT = (
    "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " + PARENT_SCRIPT
)
CHILD_SCRIPT = "import time; time.sleep(60)"


class TestTerminateAndKillProcessTree(CustomTestCase):
    def _launch(self, parent_script):
        """A parent holding one child, plus psutil handles for both."""
        process = subprocess.Popen(
            [sys.executable, "-c", parent_script, CHILD_SCRIPT],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(process.stdout.close)

        # EOF here means the parent died before reporting; fail instead of
        # blocking the CI job on a readline that never returns.
        reported_pid = process.stdout.readline()
        self.assertTrue(reported_pid, "parent exited before reporting its child pid")
        child_pid = int(reported_pid)

        def cleanup_processes():
            kill_process_tree(process.pid, wait_timeout=5)
            kill_process_tree(child_pid, wait_timeout=5)

        self.addCleanup(cleanup_processes)
        return process, psutil.Process(process.pid), psutil.Process(child_pid)

    def _assert_gone(self, proc, label):
        # A zombie has already released its resources; only a running one leaks.
        try:
            leaked = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            leaked = False
        self.assertFalse(leaked, f"{label} {proc.pid} survived the cleanup")

    def test_reaps_descendant_after_parent_exits(self):
        """Cleanup must not leave a descendant behind when the parent dies first.

        Once the parent PID is gone its children are unwalkable, so a cleanup
        that only discovers the tree after termination silently leaks them.
        """
        process, parent, child = self._launch(PARENT_SCRIPT)

        terminate_and_kill_process_tree(process, terminate_timeout=5, wait_timeout=5)

        self._assert_gone(parent, "parent")
        self._assert_gone(child, "child")

    def test_reaps_tree_when_parent_ignores_sigterm(self):
        """A parent that outlives SIGTERM still gets the whole tree SIGKILLed.

        The parent stays walkable here, so this path holds either side of the
        snapshot move; it pins the SIGKILL fallback against a later regression.
        """
        process, parent, child = self._launch(STUBBORN_PARENT_SCRIPT)

        terminate_and_kill_process_tree(process, terminate_timeout=2, wait_timeout=5)

        self._assert_gone(parent, "parent")
        self._assert_gone(child, "child")


if __name__ == "__main__":
    unittest.main()
