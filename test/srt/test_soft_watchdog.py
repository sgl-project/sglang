import io
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class BaseTestSoftWatchdog:
    env_override = None
    expected_message = None

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()

        with cls.env_override.override(5):
            cls.process = popen_launch_server(
                "Qwen/Qwen3-0.6B",
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=("--soft-watchdog-timeout", "2"),
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def test_watchdog_triggers(self):
        requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            },
            timeout=30,
        )

        time.sleep(10)

        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(self.expected_message, combined_output)


class TestSoftWatchdogDetokenizer(BaseTestSoftWatchdog):
    env_override = envs.SGLANG_TEST_STUCK_DETOKENIZER
    expected_message = "DetokenizerManager watchdog timeout"


class TestSoftWatchdogTokenizer(BaseTestSoftWatchdog):
    env_override = envs.SGLANG_TEST_STUCK_TOKENIZER
    expected_message = "TokenizerManager watchdog timeout"


if __name__ == "__main__":
    unittest.main()
