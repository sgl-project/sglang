import subprocess
import sys
import unittest

import psutil

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import terminate_and_kill_process_tree

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTerminateAndKillProcessTree(unittest.TestCase):
    def test_reaps_descendant_after_parent_exits(self):
        child_script = "import time; time.sleep(60)"
        parent_script = (
            "import subprocess, sys, time; "
            "child = subprocess.Popen("
            "[sys.executable, '-c', sys.argv[1]], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            "print(child.pid, flush=True); "
            "time.sleep(60)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", parent_script, child_script],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(process.stdout.close)

        child_pid = int(process.stdout.readline())

        def cleanup_processes():
            kill_process_tree(process.pid, wait_timeout=5)
            kill_process_tree(child_pid, wait_timeout=5)

        self.addCleanup(cleanup_processes)

        child = psutil.Process(child_pid)
        terminate_and_kill_process_tree(process, terminate_timeout=5, wait_timeout=5)

        self.assertIsNotNone(process.poll())
        self.assertFalse(child.is_running() and child.status() != psutil.STATUS_ZOMBIE)


if __name__ == "__main__":
    unittest.main()
