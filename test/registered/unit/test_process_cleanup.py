import subprocess
import sys
import unittest

import psutil

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import terminate_and_kill_process_tree

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

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
CHILD_SCRIPT = "import time; time.sleep(60)"


class TestTerminateAndKillProcessTree(unittest.TestCase):
    def test_reaps_descendant_after_parent_exits(self):
        """Cleanup must not leave a descendant behind when the parent dies first.

        Once the parent PID is gone its children are unwalkable, so a cleanup
        that only discovers the tree after termination silently leaks them.
        """
        process = subprocess.Popen(
            [sys.executable, "-c", PARENT_SCRIPT, CHILD_SCRIPT],
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

        child = psutil.Process(child_pid)
        terminate_and_kill_process_tree(process, terminate_timeout=5, wait_timeout=5)

        self.assertIsNotNone(process.poll(), "parent survived the cleanup")
        # A zombie has already released its resources; only a running child leaks.
        child_leaked = child.is_running() and child.status() != psutil.STATUS_ZOMBIE
        self.assertFalse(child_leaked, f"child {child_pid} survived the cleanup")


if __name__ == "__main__":
    unittest.main()
