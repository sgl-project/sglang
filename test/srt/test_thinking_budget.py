"""
Usage:
python3 -m unittest test_thinking_budget.TestThinkingBudget.test_chat_completion_with_thinking_budget_20
python3 -m unittest test_thinking_budget.TestThinkingBudget.test_chat_completion_with_thinking_budget_200
"""

import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestThinkingBudget(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--reasoning-parser",
                "qwen3",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_chat_completion_with_thinking_budget_20(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "9.11 and 9.8, which is greater?"}
                ],
                "temperature": 0,
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": True},
                "thinking_budget": 20,
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()
        reasoning_content = data["choices"][0]["message"]["reasoning_content"]
        tokens = self.tokenizer.encode(reasoning_content)
        self.assertEqual(
            len(tokens),
            20,
            f"Reasoning content length: {len(tokens)} not equal to 20, tokens: {tokens}, reasoning_content: {reasoning_content}",
        )

    def test_chat_completion_with_thinking_budget_200(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "9.11 and 9.8, which is greater?"}
                ],
                "temperature": 0,
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": True},
                "thinking_budget": 200,
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()
        reasoning_content = data["choices"][0]["message"]["reasoning_content"]
        tokens = self.tokenizer.encode(reasoning_content)
        self.assertEqual(
            len(tokens),
            200,
            f"Reasoning content length {len(tokens)} not equal to 200, tokens: {tokens}, reasoning_content: {reasoning_content}",
        )


if __name__ == "__main__":
    unittest.main()
