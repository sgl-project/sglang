"""
Test soft watchdog functionality for various processes.

Usage:
python -m pytest test/srt/test_soft_watchdog.py -v -s
"""

import io
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestSoftWatchdogDetokenizer(CustomTestCase):
    """Test that DetokenizerManager soft watchdog triggers on slow processing."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()

        with envs.SGLANG_TEST_WATCHDOG_SLOW_DETOKENIZER.override(5):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=(
                    "--soft-watchdog-timeout",
                    "2",
                ),
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_detokenizer_watchdog_triggers(self):
        """Send a request and verify watchdog timeout appears in logs."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            },
            timeout=30,
        )

        time.sleep(10)

        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(
            "DetokenizerManager watchdog timeout",
            combined_output,
            "Soft watchdog timeout message not found in logs",
        )


class TestSoftWatchdogTokenizer(CustomTestCase):
    """Test that TokenizerManager soft watchdog triggers on slow processing."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()

        with envs.SGLANG_TEST_WATCHDOG_SLOW_TOKENIZER.override(5):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=(
                    "--soft-watchdog-timeout",
                    "2",
                ),
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_tokenizer_watchdog_triggers(self):
        """Send a request and verify watchdog timeout appears in logs."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            },
            timeout=30,
        )

        time.sleep(10)

        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(
            "TokenizerManager watchdog timeout",
            combined_output,
            "Soft watchdog timeout message not found in logs",
        )


if __name__ == "__main__":
    unittest.main()
