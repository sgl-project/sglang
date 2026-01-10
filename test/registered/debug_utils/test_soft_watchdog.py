import io
import unittest
from contextlib import nullcontext

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)


class BaseTestSoftWatchdog:
    env_override = None
    expected_message = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    triggers_during_init = False

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        cls.process = None

        expect_timeout = unittest.TestCase().assertRaises(TimeoutError) if cls.triggers_during_init else nullcontext()
        with cls.env_override(), expect_timeout:
            cls.process = popen_launch_server(
                "Qwen/Qwen3-0.6B",
                DEFAULT_URL_FOR_TEST,
                timeout=cls.timeout,
                other_args=[
                    "--soft-watchdog-timeout",
                    "20",
                    "--skip-server-warmup",
                ],
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def test_watchdog_triggers(self):
        if not self.triggers_during_init:
            print("Start call /generate API", flush=True)
            try:
                requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "text": "Hello, please repeat this sentence for 1000 times.",
                        "sampling_params": {"max_new_tokens": 100, "temperature": 0},
                    },
                    timeout=30,
                )
            except requests.exceptions.ReadTimeout as e:
                print(f"requests.post timeout (but expected): {e}")
            print("End call /generate API", flush=True)

        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(self.expected_message, combined_output)


class TestSoftWatchdogDetokenizer(BaseTestSoftWatchdog, CustomTestCase):
    env_override = lambda: envs.SGLANG_TEST_STUCK_DETOKENIZER.override(30)
    expected_message = "DetokenizerManager watchdog timeout"


class TestSoftWatchdogTokenizer(BaseTestSoftWatchdog, CustomTestCase):
    env_override = lambda: envs.SGLANG_TEST_STUCK_TOKENIZER.override(30)
    expected_message = "TokenizerManager watchdog timeout"


class TestSoftWatchdogSchedulerInit(BaseTestSoftWatchdog, CustomTestCase):
    env_override = lambda: envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.override(30)
    expected_message = "Scheduler watchdog timeout"
    timeout = 40
    triggers_during_init = True


if __name__ == "__main__":
    unittest.main()
