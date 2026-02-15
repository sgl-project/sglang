import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestEnableThinking(CustomTestCase):
    """Testcase: Testing with the 'enable_thinking' feature enabled/disabled,
                 both streaming and non-streaming input requests successful

    [Test Category] Interface
    [Test Target] /v1/chat/completions
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        cls.other_args = [
            "--reasoning-parser",
            "qwen3",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.95,
            "--tp",
            2,
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=cls.other_args,
        )
        cls.additional_chat_kwargs = {}

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_chat_completion_with_reasoning(self):
        # Test non-streaming with "enable_thinking": True, reasoning_content should not be empty
        client = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": True},
                **self.additional_chat_kwargs,
            },
        )

        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        data = client.json()

        self.assertIn("choices", data)
        self.assertTrue(len(data["choices"]) > 0)
        self.assertIn("message", data["choices"][0])
        self.assertIn("reasoning_content", data["choices"][0]["message"])
        self.assertIsNotNone(data["choices"][0]["message"]["reasoning_content"])

    def test_chat_completion_without_reasoning(self):
        # Test non-streaming with "enable_thinking": False, reasoning_content should be empty
        client = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "separate_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": False},
                **self.additional_chat_kwargs,
            },
        )

        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        data = client.json()

        self.assertIn("choices", data)
        self.assertTrue(len(data["choices"]) > 0)
        self.assertIn("message", data["choices"][0])

        if "reasoning_content" in data["choices"][0]["message"]:
            self.assertIsNone(data["choices"][0]["message"]["reasoning_content"])

    def test_stream_chat_completion_with_reasoning(self):
        # Test streaming with "enable_thinking": True, reasoning_content should not be empty
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "separate_reasoning": True,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": True},
                **self.additional_chat_kwargs,
            },
            stream=True,
        )

        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        has_reasoning = False
        has_content = False

        print("\n=== Stream With Reasoning ===")
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})

                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            has_reasoning = True

                        if "content" in delta and delta["content"]:
                            has_content = True

        self.assertTrue(
            has_reasoning,
            "The reasoning content is not included in the stream response",
        )
        self.assertTrue(
            has_content, "The stream response does not contain normal content"
        )

    def test_stream_chat_completion_without_reasoning(self):
        # Test streaming with "enable_thinking": False, reasoning_content should  be empty
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "separate_reasoning": True,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
                **self.additional_chat_kwargs,
            },
            stream=True,
        )

        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        has_reasoning = False
        has_content = False

        print("\n=== Stream Without Reasoning ===")
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})

                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            has_reasoning = True

                        if "content" in delta and delta["content"]:
                            has_content = True

        self.assertFalse(
            has_reasoning,
            "The reasoning content should not be included in the stream response",
        )
        self.assertTrue(
            has_content, "The stream response does not contain normal content"
        )


if __name__ == "__main__":
    unittest.main()
