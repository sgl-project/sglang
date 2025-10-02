"""
python3 -m unittest test_sagemaker_server.TestSageMakerServer.test_chat_completion
"""

import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestSageMakerServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
        )
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_chat_completion(self, logprobs, parallel_sample_num):
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in a few words.",
                },
            ],
            "temperature": 0,
            "logprobs": logprobs is not None and logprobs > 0,
            "top_logprobs": logprobs,
            "n": parallel_sample_num,
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(
            f"{self.base_url}/invocations", json=data, headers=headers
        ).json()

        if logprobs:
            assert isinstance(
                response["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0][
                    "token"
                ],
                str,
            )

            ret_num_top_logprobs = len(
                response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            )
            assert (
                ret_num_top_logprobs == logprobs
            ), f"{ret_num_top_logprobs} vs {logprobs}"

        assert len(response["choices"]) == parallel_sample_num
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(response["choices"][0]["message"]["content"], str)
        assert response["id"]
        assert response["created"]
        assert response["usage"]["prompt_tokens"] > 0
        assert response["usage"]["completion_tokens"] > 0
        assert response["usage"]["total_tokens"] > 0

    def run_chat_completion_stream(self, logprobs, parallel_sample_num=1):
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in a few words.",
                },
            ],
            "temperature": 0,
            "logprobs": logprobs is not None and logprobs > 0,
            "top_logprobs": logprobs,
            "stream": True,
            "stream_options": {"include_usage": True},
            "n": parallel_sample_num,
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(
            f"{self.base_url}/invocations", json=data, stream=True, headers=headers
        )

        is_firsts = {}
        for line in response.iter_lines():
            line = line.decode("utf-8").replace("data: ", "")
            if len(line) < 1 or line == "[DONE]":
                continue
            print(f"value: {line}")
            line = json.loads(line)
            usage = line.get("usage")
            if usage is not None:
                assert usage["prompt_tokens"] > 0
                assert usage["completion_tokens"] > 0
                assert usage["total_tokens"] > 0
                continue

            index = line.get("choices")[0].get("index")
            data = line.get("choices")[0].get("delta")

            if is_firsts.get(index, True):
                assert data["role"] == "assistant"
                is_firsts[index] = False
                continue

            if logprobs:
                assert line.get("choices")[0].get("logprobs")
                assert isinstance(
                    line.get("choices")[0]
                    .get("logprobs")
                    .get("content")[0]
                    .get("top_logprobs")[0]
                    .get("token"),
                    str,
                )
                assert isinstance(
                    line.get("choices")[0]
                    .get("logprobs")
                    .get("content")[0]
                    .get("top_logprobs"),
                    list,
                )
                ret_num_top_logprobs = len(
                    line.get("choices")[0]
                    .get("logprobs")
                    .get("content")[0]
                    .get("top_logprobs")
                )
                assert (
                    ret_num_top_logprobs == logprobs
                ), f"{ret_num_top_logprobs} vs {logprobs}"

            assert isinstance(data["content"], str)
            assert line["id"]
            assert line["created"]

        for index in [i for i in range(parallel_sample_num)]:
            assert not is_firsts.get(
                index, True
            ), f"index {index} is not found in the response"

    def test_chat_completion(self):
        for logprobs in [None, 5]:
            for parallel_sample_num in [1, 2]:
                self.run_chat_completion(logprobs, parallel_sample_num)

    def test_chat_completion_stream(self):
        for logprobs in [None, 5]:
            for parallel_sample_num in [1, 2]:
                self.run_chat_completion_stream(logprobs, parallel_sample_num)


if __name__ == "__main__":
    unittest.main()
