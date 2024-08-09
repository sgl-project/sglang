import json
import os
import sys
import unittest

import requests

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, popen_launch_server

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class TestSRTEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = "http://127.0.0.1:8157"
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=300, other_args=["--skip-tokenizer-init"]
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def run_decode(
        self, return_logprob=False, top_logprobs_num=0, return_text=False, n=1
    ):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": [
                    119689,
                    50650,
                    18291,
                    30061,
                    5316,
                    26951,
                    119690,
                ],  # The capital of France is
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": 32,
                    "n": n,
                    "stop_token_ids": [119690],
                },
                "stream": False,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "return_text_in_logprobs": return_text,
                "logprob_start_len": 0,
            },
        )
        print(json.dumps(response.json()))
        print("=" * 100)

    def test_simple_decode(self):
        self.run_decode()

    def test_parallel_sample(self):
        self.run_decode(n=3)

    def test_logprob(self):
        for top_logprobs_num in [0, 3]:
            for return_text in [False, False]:
                self.run_decode(
                    return_logprob=True,
                    top_logprobs_num=top_logprobs_num,
                    return_text=return_text,
                )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
