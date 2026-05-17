"""
Hidden states extraction test for Intel XPU.

Tests hidden states API to verify that intermediate layer outputs
can be extracted correctly on XPU.

Based on test/registered/core/test_hidden_states.py
adapted for XPU Stage B (proven passing in HTML report).

Usage:
python3 -m unittest test_hidden_states_xpu.TestHiddenStatesXPU
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=45, suite="stage-b-test-1-gpu-xpu")


class TestHiddenStatesXPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--device",
                "xpu",
                "--mem-fraction-static",
                "0.7",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_hidden_states_basic(self):
        """Test basic hidden states extraction."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                },
                "return_hidden_states": True,
            },
            timeout=30,
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("hidden_states", result)
        self.assertIsNotNone(result["hidden_states"])
        self.assertGreater(len(result["hidden_states"]), 0)


if __name__ == "__main__":
    unittest.main()
