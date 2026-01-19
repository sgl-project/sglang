import json
import unittest
import requests
import os
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestReasoningOffByDefault(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"

        # Set the environment variable for the server process
        cls.env = os.environ.copy()
        cls.env["SGLANG_REASONING_OFF_BY_DEFAULT"] = "True"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--reasoning-parser",
                "qwen3",
            ],
            env=cls.env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_reasoning_is_off_by_default(self):
        # Test that reasoning is OFF when no chat_template_kwargs are passed
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "temperature": 0,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        message = data["choices"][0]["message"]
        self.assertNotIn("reasoning_content", message)

    def test_reasoning_can_be_enabled_explicitly(self):
        # Test that reasoning can still be enabled explicitly
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "temperature": 0,
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        message = data["choices"][0]["message"]
        self.assertIn("reasoning_content", message)
        self.assertIsNotNone(message["reasoning_content"])

    def test_reasoning_off_by_default_without_parser(self):
        # Test that reasoning is OFF when no reasoning parser is configured
        # but SGLANG_REASONING_OFF_BY_DEFAULT is set
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "temperature": 0,
                "separate_reasoning": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        message = data["choices"][0]["message"]
        # When reasoning is off by default, there should be no reasoning_content
        self.assertNotIn("reasoning_content", message)


if __name__ == "__main__":
    unittest.main()
