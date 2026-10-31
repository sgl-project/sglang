import os
import subprocess
import sys
import unittest
from unittest import mock

from sglang.cli.killall import _find_sglang_pids_by_name
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@unittest.skipUnless(os.path.isdir("/proc"), "requires /proc")
class TestKillallRunnerScope(unittest.TestCase):
    def _spawn_fake_server(self, runner_name):
        env = {**os.environ, "RUNNER_NAME": runner_name}
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(60)",
                "sglang.launch_server",
            ],
            env=env,
        )
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        return proc.pid

    def test_skips_other_runners_processes(self):
        """Runners sharing one PID namespace must not kill each other's servers
        at job start; orphans from this runner's own jobs are still found."""
        own = self._spawn_fake_server("runner-a")
        other = self._spawn_fake_server("runner-b")
        with mock.patch.dict(os.environ, {"RUNNER_NAME": "runner-a"}):
            pids = _find_sglang_pids_by_name()
        self.assertIn(own, pids)
        self.assertNotIn(other, pids)


if __name__ == "__main__":
    unittest.main()
