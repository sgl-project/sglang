"""
Usage:
python3 -m unittest openai_server.features.test_usage_tokens.TestReasoningTokenUsage
"""

import json
import unittest

from openai import OpenAI

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_REASONING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


class TestNormalDecoding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_REASONING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        reasoning_parser_name = "deepseek-r1"

        # get think_end_token_id
        cls.tokenizer = get_tokenizer(cls.model)
        reasoning_parser = ReasoningParser(reasoning_parser_name)
        cls.think_end_token_id = cls.tokenizer.convert_tokens_to_ids(
            reasoning_parser.detector.think_end_token
        )
        assert cls.think_end_token_id, "think_end_token_id shouldn't be None"

        # launch_server & create client
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=["--reasoning-parser", reasoning_parser_name],
        )
        cls.client = OpenAI(base_url=f"{cls.base_url}/v1", api_key=cls.api_key)
        cls.messages = [{"role": "user", "content": "What is 1+3?"}]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_generate_api_non_streaming(self):
        import requests

        response = requests.post(
            url=f"{self.base_url}/generate",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "text": self.tokenizer.apply_chat_template(
                    self.messages, add_generation_prompt=True, tokenize=False
                ),
                "model": self.model,
                "require_reasoning": True,
                "sampling_params": {"max_new_tokens": 1024},
            },
        )
        response.raise_for_status()
        res_json = response.json()
        report_reasoning_tokens = res_json["meta_info"]["reasoning_tokens"]
        actual_reasoning_tokens = (
            res_json["output_ids"].index(self.think_end_token_id) + 1
        )
        self.assertEqual(report_reasoning_tokens, actual_reasoning_tokens)

    def test_generate_api_streaming(self):
        import requests

        response = requests.post(
            url=f"{self.base_url}/generate",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "text": self.tokenizer.apply_chat_template(
                    self.messages, add_generation_prompt=True, tokenize=False
                ),
                "model": self.model,
                "require_reasoning": True,
                "sampling_params": {"max_new_tokens": 1024},
                "stream": True,
            },
            stream=True,
        )
        response.raise_for_status()
        for chunk in response.iter_lines():
            if not chunk:
                continue
            decoded_str = remove_prefix(chunk.decode("utf-8"), "data: ")
            if decoded_str == "[DONE]":
                pass
            else:
                data = json.loads(decoded_str)
                report_reasoning_tokens = data["meta_info"]["reasoning_tokens"]
                if self.think_end_token_id in data["output_ids"]:
                    actual_reasoning_tokens = (
                        data["output_ids"].index(self.think_end_token_id) + 1
                    )
                else:
                    actual_reasoning_tokens = len(data["output_ids"])
                self.assertEqual(report_reasoning_tokens, actual_reasoning_tokens)

    def test_chat_api_non_streaming(self):
        payload = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": 1024,
        }
        response = self.client.chat.completions.create(**payload)

        assert response.usage is not None
        self.assertNotEqual(response.usage.reasoning_tokens, 0)

    def test_chat_api_streaming(self):
        payload = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": 1024,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
        response = self.client.chat.completions.create(**payload)

        for chunk in response:
            if chunk.usage:
                self.assertNotEqual(chunk.usage.reasoning_tokens, 0)


if __name__ == "__main__":
    unittest.main()
