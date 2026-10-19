"""Exercise server-exit handling with real subprocesses, without a model or GPU."""

import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_CHILD_SCRIPT = textwrap.dedent(
    """
    import subprocess
    import sys
    import time
    import unittest

    from sglang.test.test_utils import CustomTestCase

    scenario = sys.argv[1]

    class ServerExitCase(CustomTestCase):
        expect_server_exit = scenario.startswith("expected")
        attempts = 0

        def stop_server(self):
            self.process.communicate(input="exit", timeout=10)
            self.assertEqual(self.process.returncode, 7)

        def test_exit(self):
            self.attempts += 1
            print(f"ATTEMPT={self.attempts}", flush=True)
            if scenario == "teardown":
                return
            self.stop_server()
            # Keep the method active long enough for the watcher to observe exit.
            time.sleep(10 if scenario == "unexpected" else 1)
            if scenario == "expected_failure":
                self.fail("expected assertion failure")
            print("ASSERTIONS_COMPLETED", flush=True)

        def tearDown(self):
            if scenario == "teardown":
                self.stop_server()
                time.sleep(1)
                print("TEARDOWN_COMPLETED", flush=True)

    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read(); sys.exit(7)"],
        stdin=subprocess.PIPE,
        text=True,
    ) as process:
        case = ServerExitCase("test_exit")
        case.process = process
        # Run the method lifecycle directly; no model setup or CI GPU-idle check.
        result = unittest.TextTestRunner().run(case)
    sys.exit(0 if result.wasSuccessful() else 1)
    """
)


class TestCustomTestCaseServerExit(unittest.TestCase):
    def run_scenario(self, scenario):
        env = os.environ.copy()
        env["SGLANG_TEST_MAX_RETRY"] = "2"
        return subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT, scenario],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_unexpected_exit_aborts_runner(self):
        result = self.run_scenario("unexpected")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertIn("[FIXTURE FAIL-FAST]", result.stderr)
        self.assertNotIn("ASSERTIONS_COMPLETED", result.stdout)

    def test_expected_exit_reaches_assertions(self):
        result = self.run_scenario("expected")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ASSERTIONS_COMPLETED", result.stdout)
        self.assertNotIn("[FIXTURE FAIL-FAST]", result.stderr)

    def test_expected_exit_does_not_retry_failed_assertion(self):
        result = self.run_scenario("expected_failure")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("expected assertion failure", result.stderr)
        self.assertEqual(result.stdout.count("ATTEMPT="), 1, result.stdout)
        self.assertNotIn("[FIXTURE FAIL-FAST]", result.stderr)

    def test_teardown_exit_does_not_abort_runner(self):
        result = self.run_scenario("teardown")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TEARDOWN_COMPLETED", result.stdout)
        self.assertNotIn("[FIXTURE FAIL-FAST]", result.stderr)


if __name__ == "__main__":
    unittest.main()
