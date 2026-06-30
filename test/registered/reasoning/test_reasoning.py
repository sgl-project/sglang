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

    @staticmethod
    def _responses_output_text(data):
        return "".join(
            content["text"]
            for item in data["output"]
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        )

    def assert_responses_has_no_reasoning_item(self, data):
        self.assertFalse(
            any(item.get("type") == "reasoning" for item in data["output"]),
            data["output"],
        )

    def test_responses_chat_template_kwargs_disable_reasoning(self):
        response = requests.post(
            f"{self.base_url}/v1/responses",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": "What is 123 + 456? Explain briefly.",
                "temperature": 0,
                "max_output_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()
        self.assert_responses_has_no_reasoning_item(data)
        self.assertTrue(self._responses_output_text(data))

    def test_responses_structured_output_matches_chat_and_disables_reasoning(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        }
        prompt = "Invent a random fictional character and tell me a little about them."

        chat_resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 128,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "person",
                        "strict": True,
                        "schema": schema,
                    },
                },
                "chat_template_kwargs": {"enable_thinking": False},
                **self.additional_chat_kwargs,
            },
        )
        self.assertEqual(chat_resp.status_code, 200, f"Failed with: {chat_resp.text}")
        chat_text = chat_resp.json()["choices"][0]["message"]["content"]
        chat_obj = json.loads(chat_text)
        self.assertIsInstance(chat_obj["name"], str)
        self.assertIsInstance(chat_obj["age"], int)

        responses_resp = requests.post(
            f"{self.base_url}/v1/responses",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": prompt,
                "temperature": 0,
                "max_output_tokens": 128,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "person",
                        "strict": True,
                        "schema": schema,
                    }
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.assertEqual(
            responses_resp.status_code, 200, f"Failed with: {responses_resp.text}"
        )
        data = responses_resp.json()
        self.assert_responses_has_no_reasoning_item(data)
        output_text = self._responses_output_text(data)
        responses_obj = json.loads(output_text)
        self.assertIsInstance(responses_obj["name"], str)
        self.assertIsInstance(responses_obj["age"], int)

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
