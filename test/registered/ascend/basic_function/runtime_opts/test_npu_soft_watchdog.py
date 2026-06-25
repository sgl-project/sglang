import io
import logging
import os
import unittest
from contextlib import contextmanager

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Initialize logging configuration (replace print)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Register CI task for NPU environment
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

# ===================== Common Env Var Management Functions =====================
TEST_RELATED_ENVS = ["SGLANG_IS_IN_CI", "SGLANG_TEST_STUCK_DETOKENIZER"]


@contextmanager
def temporary_test_envs(ci_mode: bool = None, stuck_detokenizer: int = None):
    """Manage only test-related environment variables, auto save/restore"""
    original_values = {var: os.environ.get(var) for var in TEST_RELATED_ENVS}
    try:
        if ci_mode is not None:
            os.environ["SGLANG_IS_IN_CI"] = "True" if ci_mode else "False"
        if stuck_detokenizer is not None:
            os.environ["SGLANG_TEST_STUCK_DETOKENIZER"] = str(stuck_detokenizer)
        yield
    finally:
        for var_name, original_val in original_values.items():
            if original_val is None:
                os.environ.pop(var_name, None)
            else:
                os.environ[var_name] = original_val


class BaseTestDetokenizerWatchdog:
    """Testcase: Ensure that soft-watchdog-timeout is set by default in the CI environment, and in non-CI environments it is not set by default and needs to be set manually.

    [Test Category] Parameter
    [Test Target] --soft-watchdog-timeout
    """

    ci_mode = None
    set_soft_watchdog = None
    soft_watchdog_value = 10
    stuck_seconds = 350
    expected_log = None
    expected_assert_error = (
        "stuck tester can be enabled only if soft watchdog is enabled"
    )

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        cls.process = None
        cls.launch_success = False
        cls.error_found_in_log = False  # Mark if expected error is found in logs

        # Build launch arguments (whether to set soft-watchdog-timeout)
        other_args = ["--skip-server-warmup"]
        if cls.set_soft_watchdog:
            other_args.extend(["--soft-watchdog-timeout", str(cls.soft_watchdog_value)])

        # Scenario 4 timeout set to 20 seconds (ensure complete log printing)
        timeout = (
            20
            if (cls.ci_mode is False and cls.set_soft_watchdog is False)
            else DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

        try:
            # Simulate detokenizer blocking
            with envs.SGLANG_TEST_STUCK_DETOKENIZER.override(cls.stuck_seconds):
                cls.process = popen_launch_server(
                    QWEN3_0_6B_WEIGHTS_PATH,
                    DEFAULT_URL_FOR_TEST,
                    timeout=timeout,
                    other_args=other_args,
                    return_stdout_stderr=(cls.stdout, cls.stderr),
                )
            cls.launch_success = True
        except TimeoutError:
            # Scenario 4 expects timeout, check if target error exists in logs
            cls.launch_success = False
            # Read complete logs
            combined_log = cls.stdout.getvalue() + cls.stderr.getvalue()
            # Check if contains expected AssertionError string
            if cls.expected_assert_error in combined_log:
                cls.error_found_in_log = True
                logger.info(
                    f"\n[Scenario 4] Found expected error in logs: {cls.expected_assert_error}"
                )
                # Print complete logs for troubleshooting
                logger.info(f"\n[Scenario 4] Complete logs:\n{combined_log}")
            else:
                # Expected error not found, raise timeout error
                raise

    @classmethod
    def tearDownClass(cls):
        # Final fallback cleanup
        if cls.process:
            kill_process_tree(cls.process.pid)
        if cls.stdout:
            cls.stdout.close()
        if cls.stderr:
            cls.stderr.close()

    def test_detokenizer_watchdog(self):
        # Scenario 4: Non-CI + no soft watchdog → verify AssertionError in logs
        if self.ci_mode is False and self.set_soft_watchdog is False:
            self.assertTrue(
                self.error_found_in_log,
                f"Scenario 4: Expected error not found in logs: {self.expected_assert_error}",
            )
            logger.info(
                "[Scenario 4] Test passed: Found expected AssertionError string in logs"
            )
            return

        # Scenarios 1-3: Launch success → call API and verify timeout logs
        self.assertTrue(self.launch_success, "Server launch failed")
        logger.info("Start call /generate API", extra={"flush": True})
        requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello, please repeat this sentence for 1000 times.",
                "sampling_params": {"max_new_tokens": 100, "temperature": 0},
            },
            timeout=40,
        )
        logger.info("Start call /generate API", extra={"flush": True})

        # Merge output and verify expected logs
        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(
            self.expected_log,
            combined_output,
            f"Expected log not found: {self.expected_log}",
        )
        logger.info(
            f"[Scenario {self.__class__.__name__}] Test passed: Found expected log {self.expected_log}"
        )


# ===================== Test Subclasses for Four Scenarios =====================
# Scenario 1: CI environment + no soft-watchdog (default 300s) → block 350s
class TestCIWithoutSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = True
    set_soft_watchdog = False
    stuck_seconds = 350
    expected_log = "DetokenizerManager watchdog timeout"


# Scenario 2: CI environment + set soft-watchdog (20s) → block 30s
class TestCIWithSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = True
    set_soft_watchdog = True
    soft_watchdog_value = 20
    stuck_seconds = 30
    expected_log = "DetokenizerManager watchdog timeout"


# Scenario 3: Non-CI environment + set soft-watchdog (20s) → block 30s
class TestNonCIWithSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = False
    set_soft_watchdog = True
    soft_watchdog_value = 20
    stuck_seconds = 30
    expected_log = "DetokenizerManager watchdog timeout"


# Scenario 4: Non-CI environment + no soft-watchdog (verify AssertionError in logs)
class TestNonCIWithoutSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = False
    set_soft_watchdog = False


# ===================== Test Execution Function =====================
def run_test_scenario(test_case_cls):
    """Run single test scenario, auto manage environment variables"""
    with temporary_test_envs(ci_mode=test_case_cls.ci_mode):
        suite = unittest.TestLoader().loadTestsFromTestCase(test_case_cls)
        unittest.TextTestRunner(verbosity=2).run(suite)


# ===================== Main Function (Execute Four Scenarios) =====================
if __name__ == "__main__":
    # Scenario 1: CI + no soft-watchdog
    logger.info("=== Scenario 1: CI Environment - No soft-watchdog ===")
    run_test_scenario(TestCIWithoutSoftWatchdog)

    # Scenario 2: CI + set soft-watchdog(20s) → block 30s
    logger.info(
        "\n=== Scenario 2: CI Environment - Set soft-watchdog(20s), block 30s ==="
    )
    run_test_scenario(TestCIWithSoftWatchdog)

    # Scenario 3: Non-CI + set soft-watchdog(20s) → block 30s
    logger.info(
        "\n=== Scenario 3: Non-CI Environment - Set soft-watchdog(20s), block 30s ==="
    )
    run_test_scenario(TestNonCIWithSoftWatchdog)

    # Scenario 4: Non-CI + no soft-watchdog (verify AssertionError)
    logger.info(
        "\n=== Scenario 4: Non-CI Environment - No soft-watchdog (Verify AssertionError) ==="
    )
    run_test_scenario(TestNonCIWithoutSoftWatchdog)
