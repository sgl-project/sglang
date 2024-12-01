"""
python3 -m unittest test_skip_tokenizer_init.TestSkipTokenizerInit.test_parallel_sample
"""

import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestSkipTokenizerInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--skip-tokenizer-init"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, return_logprob=False, top_logprobs_num=0, n=1):
        max_new_tokens = 32
        input_ids = [128000, 791, 6864, 315, 9822, 374]  # The capital of France is
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": max_new_tokens,
                    "n": n,
                    "stop_token_ids": [119690],
                },
                "stream": False,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "logprob_start_len": 0,
            },
        )
        ret = response.json()
        print(json.dumps(ret))

        def assert_one_item(item):
            assert len(item["token_ids"]) == item["meta_info"]["completion_tokens"]
            assert len(item["token_ids"]) == max_new_tokens
            assert item["meta_info"]["prompt_tokens"] == len(input_ids)

            if return_logprob:
                assert len(item["meta_info"]["input_token_logprobs"]) == len(
                    input_ids
                ), f'{len(item["meta_info"]["input_token_logprobs"])} vs. f{len(input_ids)}'
                assert len(item["meta_info"]["output_token_logprobs"]) == max_new_tokens

        if n == 1:
            assert_one_item(ret)
        else:
            assert len(ret) == n
            for i in range(n):
                assert_one_item(ret[i])

        print("=" * 100)

    def test_simple_decode(self):
        self.run_decode()

    def test_parallel_sample(self):
        self.run_decode(n=3)

    def test_logprob(self):
        for top_logprobs_num in [0, 3]:
            self.run_decode(
                return_logprob=True,
                top_logprobs_num=top_logprobs_num,
            )


if __name__ == "__main__":
    unittest.main()
