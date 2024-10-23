"""
python3 -m unittest test_srt_endpoint.TestSRTEndpoint.test_simple_decode
"""

import json
import unittest

import requests

from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestSRTEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def run_decode(
        self,
        return_logprob=False,
        top_logprobs_num=0,
        return_text=False,
        n=1,
        stream=False,
    ):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": 16,
                    "n": n,
                },
                "stream": stream,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "return_text_in_logprobs": return_text,
                "logprob_start_len": 0,
            },
        )
        if not stream:
            response_json = response.json()
        else:
            response_json = []
            for line in response.iter_lines():
                if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                    response_json.append(json.loads(line[6:]))

        print(json.dumps(response_json, indent=2))
        print("=" * 100)

    def test_simple_decode(self):
        self.run_decode()

    def test_parallel_sample(self):
        self.run_decode(n=3)

    def test_parallel_sample_stream(self):
        self.run_decode(n=3, stream=True)

    def test_logprob(self):
        self.run_decode(
            return_logprob=True,
            top_logprobs_num=5,
            return_text=True,
        )

    def test_logprob_start_len(self):
        logprob_start_len = 4
        new_tokens = 4
        prompts = [
            "I have a very good idea on",
            "Today is a sunndy day and",
        ]

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": new_tokens,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
                "return_text_in_logprobs": True,
                "logprob_start_len": logprob_start_len,
            },
        )
        response_json = response.json()
        print(json.dumps(response_json, indent=2))

        for i, res in enumerate(response_json):
            assert res["meta_info"]["prompt_tokens"] == logprob_start_len + 1 + len(
                res["meta_info"]["input_token_logprobs"]
            )
            assert prompts[i].endswith(
                "".join([x[-1] for x in res["meta_info"]["input_token_logprobs"]])
            )

            assert res["meta_info"]["completion_tokens"] == new_tokens
            assert len(res["meta_info"]["output_token_logprobs"]) == new_tokens
            res["text"] == "".join(
                [x[-1] for x in res["meta_info"]["output_token_logprobs"]]
            )

    def test_get_memory_pool_size(self):
        response = requests.post(self.base_url + "/get_memory_pool_size")
        assert isinstance(response.json(), int)


if __name__ == "__main__":
    unittest.main()
