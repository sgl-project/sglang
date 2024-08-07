import json
import unittest

import requests

from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, popen_launch_server


class TestBatchPenalizerE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = f"http://localhost:{8157}"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=300,
            other_args=(
                "--random-seed",
                "0",
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def run_decode(
        self,
        return_logprob=True,
        top_logprobs_num=5,
        return_text=True,
        n=1,
        **sampling_params,
    ):
        response = requests.post(
            self.base_url + "/generate",
            json={
                # prompt that is supposed to generate < 32 tokens
                "text": "<|start_header_id|>user<|end_header_id|>\n\nWhat is the answer for 1 + 1 = ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "sampling_params": {
                    "max_new_tokens": 32,
                    "n": n,
                    **sampling_params,
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

    def test_default_values(self):
        self.run_decode()

    def test_frequency_penalty(self):
        self.run_decode(frequency_penalty=2)

    def test_min_new_tokens(self):
        self.run_decode(min_new_tokens=16)

    def test_presence_penalty(self):
        self.run_decode(presence_penalty=2)

    def test_repetition_penalty(self):
        self.run_decode(repetition_penalty=2)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
