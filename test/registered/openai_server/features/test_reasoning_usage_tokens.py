"""Usage:
python3 -m unittest openai_server.features.test_reasoning_usage_tokens.TestNormalReasoningTokenUsage
python3 -m unittest openai_server.features.test_reasoning_usage_tokens.TestSpecReasoningTokenUsage
python3 -m unittest openai_server.features.test_reasoning_usage_tokens.TestSpecV2ReasoningTokenUsage
"""

import json
import os
import unittest

import requests
from openai import OpenAI

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_REASONING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=90, suite="stage-b-test-large-1-gpu")
register_amd_ci(est_time=90, suite="stage-b-test-large-1-gpu-amd")


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


class ReasoningTokenUsageMixin:
    model = ""
    reasoning_parser_name = ""
    extra_server_args = []
    extra_env_vars = {}
    max_new_tokens = 1024

    @classmethod
    def setUpClass(cls):
        for k, v in cls.extra_env_vars.items():
            os.environ[k] = v

        assert cls.model
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"

        # get think_end_token_id
        cls.tokenizer = get_tokenizer(cls.model)
        reasoning_parser = ReasoningParser(cls.reasoning_parser_name)
        cls.think_end_token_id = cls.tokenizer.convert_tokens_to_ids(
            reasoning_parser.detector.think_end_token
        )
        assert (
            cls.think_end_token_id
        ), f"think_end_token_id for {cls.reasoning_parser_name} shouldn't be None"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--reasoning-parser",
                cls.reasoning_parser_name,
            ]
            + cls.extra_server_args,
        )
        cls.client = OpenAI(base_url=f"{cls.base_url}/v1", api_key=cls.api_key)
        cls.messages = [{"role": "user", "content": "What is 1+3?"}]

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_generate_api_non_streaming(self):
        response = requests.post(
            url=f"{self.base_url}/generate",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "text": self.tokenizer.apply_chat_template(
                    self.messages, add_generation_prompt=True, tokenize=False
                ),
                "model": self.model,
                "require_reasoning": True,
                "sampling_params": {"max_new_tokens": self.max_new_tokens},
            },
        )
        response.raise_for_status()
        res_json = response.json()
        report_reasoning_tokens = res_json["meta_info"]["reasoning_tokens"]
        actual_reasoning_tokens = (
            res_json["output_ids"].index(self.think_end_token_id) + 1
        )
        assert (
            report_reasoning_tokens == actual_reasoning_tokens
        ), f"Expected {actual_reasoning_tokens}, got {report_reasoning_tokens}"

    def test_generate_api_streaming(self):
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
            if decoded_str != "[DONE]":
                data = json.loads(decoded_str)
                report_reasoning_tokens = data["meta_info"]["reasoning_tokens"]
                if self.think_end_token_id in data["output_ids"]:
                    actual_reasoning_tokens = (
                        data["output_ids"].index(self.think_end_token_id) + 1
                    )
                else:
                    actual_reasoning_tokens = len(data["output_ids"])
                assert report_reasoning_tokens == actual_reasoning_tokens

    def test_chat_api_non_streaming(self):
        response = self.client.chat.completions.create(
            model=self.model, messages=self.messages, max_tokens=1024
        )
        assert response.usage is not None
        assert response.usage.reasoning_tokens > 0

    def test_chat_api_streaming(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=1024,
            stream=True,
            stream_options={"include_usage": True, "continuous_usage_stats": True},
        )
        for chunk in response:
            if chunk.usage:
                assert chunk.usage.reasoning_tokens > 0


class TestNormalReasoningTokenUsage(ReasoningTokenUsageMixin, CustomTestCase):
    model = DEFAULT_REASONING_MODEL_NAME_FOR_TEST
    reasoning_parser_name = "deepseek-r1"
    extra_server_args = ["--cuda-graph-max-bs", "2"]


class TestSpecReasoningTokenUsage(ReasoningTokenUsageMixin, CustomTestCase):
    model = "Qwen/Qwen3-30B-A3B"  # select this model due to its suitable eagle model
    reasoning_parser_name = "qwen3"
    extra_env_vars = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}
    extra_server_args = [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        "nex-agi/SGLANG-EAGLE3-Qwen3-30B-A3B-Nex-N1",
        "--cuda-graph-max-bs",
        "2",
    ]


class TestSpecV2ReasoningTokenUsage(TestSpecReasoningTokenUsage):
    extra_env_vars = {
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
    }


if __name__ == "__main__":
    unittest.main()
