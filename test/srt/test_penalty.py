import json
import random
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestPenalty(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, sampling_params):
        return_logprob = True
        top_logprobs_num = 5
        return_text = True
        n = 1

        response = requests.post(
            self.base_url + "/generate",
            json={
                # prompt that is supposed to generate < 32 tokens
                "text": "<|start_header_id|>user<|end_header_id|>\n\nWhat is the answer for 1 + 1 = ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "sampling_params": {
                    "max_new_tokens": 48,
                    "n": n,
                    **sampling_params,
                },
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "return_text_in_logprobs": return_text,
                "logprob_start_len": 0,
            },
        )
        self.assertEqual(response.status_code, 200)
        print(json.dumps(response.json()))
        print("=" * 100)

    def test_default_values(self):
        self.run_decode({})

    def test_frequency_penalty(self):
        self.run_decode({"frequency_penalty": 2})

    def test_min_new_tokens(self):
        self.run_decode({"min_new_tokens": 16})

    def test_presence_penalty(self):
        self.run_decode({"presence_penalty": 2})

    def test_penalty_mixed(self):
        args = [
            {},
            {},
            {},
            {"frequency_penalty": 2},
            {"presence_penalty": 1},
            {"min_new_tokens": 16},
            {"frequency_penalty": 0.2},
            {"presence_penalty": 0.4},
            {"min_new_tokens": 8},
            {"frequency_penalty": 0.4, "presence_penalty": 0.8},
            {"frequency_penalty": 0.4, "min_new_tokens": 12},
            {"presence_penalty": 0.8, "min_new_tokens": 12},
            {"presence_penalty": -0.3, "frequency_penalty": 1.3, "min_new_tokens": 32},
            {"presence_penalty": 0.3, "frequency_penalty": -1.3, "min_new_tokens": 32},
        ]
        random.shuffle(args * 5)
        with ThreadPoolExecutor(8) as executor:
            list(executor.map(self.run_decode, args))


if __name__ == "__main__":
    unittest.main(verbosity=3)
