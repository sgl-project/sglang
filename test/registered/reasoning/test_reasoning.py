import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.reasoning_kit import (
    ReasoningTokenUsageMixin,
    SeparateReasoningMixin,
)
from sglang.test.test_utils import (
    DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=129, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=200, suite="stage-b-test-1-gpu-small-amd")


class TestEnableThinking(
    ReasoningTokenUsageMixin, SeparateReasoningMixin, CustomTestCase
):
    reasoning_parser_name = "qwen3"

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        cls.init_reasoning_token_verifier()
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

    def test_anthropic_messages_without_thinking_blocks(self):
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": "Solve carefully but return only the final answer: What is 27 * 14?",
                    }
                ],
                "thinking": {"type": "disabled"},
            },
        )

        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()

        self.assertIn("content", data)
        self.assertTrue(len(data["content"]) > 0)
        self.assertTrue(all(block["type"] != "thinking" for block in data["content"]))
        self.assertTrue(
            all(block["type"] != "redacted_thinking" for block in data["content"])
        )
        text_blocks = [block for block in data["content"] if block["type"] == "text"]
        self.assertTrue(len(text_blocks) > 0)
        combined_text = "".join(block["text"] for block in text_blocks).lower()
        self.assertNotIn("<think", combined_text)
        self.assertNotIn("</think>", combined_text)
        self.assertNotIn("thinking process", combined_text)

    def test_anthropic_messages_stream_without_thinking_events(self):
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": "Solve carefully but return only the final answer: What is 39 + 48?",
                    }
                ],
                "thinking": {"type": "disabled"},
                "stream": True,
            },
            stream=True,
        )

        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        event_types = []
        delta_types = []
        has_text_delta = False
        text_deltas = []

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("event: "):
                event_types.append(line[7:])
                continue
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue

            payload = json.loads(line[6:])
            delta = payload.get("delta")
            if delta:
                delta_type = delta.get("type")
                delta_types.append(delta_type)
                if delta_type == "text_delta" and delta.get("text"):
                    has_text_delta = True
                    text_deltas.append(delta["text"])

        self.assertTrue(has_text_delta)
        self.assertNotIn("thinking_delta", delta_types)
        self.assertNotIn("signature_delta", delta_types)
        combined_text = "".join(text_deltas).lower()
        self.assertNotIn("<think", combined_text)
        self.assertNotIn("</think>", combined_text)
        self.assertNotIn("thinking process", combined_text)

    def test_anthropic_messages_with_thinking_blocks(self):
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2 + 2? Return only the final answer.",
                    }
                ],
                "thinking": {"type": "enabled"},
                "temperature": 0,
            },
        )

        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()

        self.assertIn("content", data)
        thinking_blocks = [
            block for block in data["content"] if block.get("type") == "thinking"
        ]
        text_blocks = [
            block for block in data["content"] if block.get("type") == "text"
        ]

        self.assertTrue(len(thinking_blocks) > 0)
        self.assertTrue(
            any(block.get("thinking") for block in thinking_blocks),
            "Expected non-empty Anthropic thinking blocks",
        )
        if text_blocks:
            self.assertTrue(any(block.get("text") for block in text_blocks))

    def test_anthropic_messages_stream_with_thinking_events(self):
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2 + 2? Return only the final answer.",
                    }
                ],
                "thinking": {"type": "enabled"},
                "temperature": 0,
                "stream": True,
            },
            stream=True,
        )

        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        delta_types = []
        has_thinking_delta = False

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue

            payload = json.loads(line[6:])
            delta = payload.get("delta")
            if delta:
                delta_type = delta.get("type")
                delta_types.append(delta_type)
                if delta_type == "thinking_delta" and delta.get("thinking"):
                    has_thinking_delta = True

        self.assertTrue(has_thinking_delta)
        self.assertIn("thinking_delta", delta_types)
        self.assertIn("signature_delta", delta_types)

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
        # Test streaming with "enable_thinking": False, reasoning_content should be empty
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
