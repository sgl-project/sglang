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
"""Tests that the CI suite runner cannot stall after results are reported.

A ROCm 7.2.4 job stayed alive for 34 minutes after its suite had already
printed `Ran 2 tests ... OK`, producing no further output until the runner was
reclaimed. A suite that has reported every result must exit, and a leftover
worker must never be able to hold the interpreter open.
"""

import subprocess
import sys
import threading
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.ci.ci_utils import run_with_timeout
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# Mirrors the tail of run_suite.main(): a worker that will never finish, a
# result line already written, then the exit path under test.
_EXIT_SCRIPT = """
import os, sys, threading
threading.Thread(target=threading.Event().wait, daemon=False).start()
print("summary line")
{exit_stmt}
"""

_HANG_BUDGET_S = 5.0


class TestRunWithTimeout(CustomTestCase):
    def setUp(self):
        self._release = threading.Event()

    def tearDown(self):
        # Let any worker abandoned by a timeout finish, so it does not leak
        # into the rest of this process.
        self._release.set()

    def test_returns_value_when_within_budget(self):
        self.assertEqual(run_with_timeout(lambda: 7, timeout=10), 7)

    def test_timed_out_worker_is_daemonized(self):
        """A worker abandoned on timeout must not outrank interpreter shutdown.

        The runner kills the child and moves on, but a child stuck in D-state
        on a wedged GPU keeps the worker blocked forever. As a non-daemon
        thread it would then hang the whole suite at exit.
        """
        with self.assertRaises(TimeoutError):
            run_with_timeout(self._release.wait, timeout=0.2)

        workers = [t for t in threading.enumerate() if t.name == "ci-file-runner"]
        self.assertEqual(len(workers), 1, "expected exactly one abandoned worker")
        self.assertTrue(workers[0].daemon, "abandoned worker must be a daemon")


class TestSuiteRunnerExitPath(CustomTestCase):
    def _run_exit_script(self, exit_stmt: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", _EXIT_SCRIPT.format(exit_stmt=exit_stmt)],
            capture_output=True,
            text=True,
            timeout=_HANG_BUDGET_S,
        )

    def test_hard_exit_terminates_and_keeps_output(self):
        proc = self._run_exit_script(
            "sys.stdout.flush(); sys.stderr.flush(); os._exit(0)"
        )
        self.assertEqual(proc.returncode, 0)
        # os._exit bypasses Python's own flush, so the explicit flush above is
        # the only thing keeping the summary out of the void.
        self.assertIn("summary line", proc.stdout)

    def test_hard_exit_reports_failure_as_255(self):
        proc = self._run_exit_script(
            "sys.stdout.flush(); sys.stderr.flush(); os._exit(-1 & 0xFF)"
        )
        self.assertEqual(proc.returncode, 255)

    def test_sys_exit_would_hang(self):
        """Guards the guard: proves the leftover worker really does block exit.

        Without this, the tests above would still pass if a stuck thread had
        never been able to stall shutdown in the first place.
        """
        with self.assertRaises(subprocess.TimeoutExpired):
            self._run_exit_script("sys.exit(0)")


if __name__ == "__main__":
    unittest.main()
