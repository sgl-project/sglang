import json
import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

_server_process = None
_base_url = None
_tokenizer = None


def setUpModule():
    """
    Launch the server once before all tests and initialize the tokenizer.
    """
    global _server_process, _base_url, _tokenizer
    _server_process = popen_launch_server(
        DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=["--skip-tokenizer-init"],
    )
    _base_url = DEFAULT_URL_FOR_TEST

    _tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_SMALL_MODEL_NAME_FOR_TEST, use_fast=False
    )
    print(">>> setUpModule: Server launched, tokenizer ready")


def tearDownModule():
    """
    Terminate the server once after all tests have completed.
    """
    global _server_process
    if _server_process is not None:
        kill_process_tree(_server_process.pid)
        _server_process = None
    print(">>> tearDownModule: Server terminated")


class TestSkipTokenizerInit(unittest.TestCase):
    def run_decode(
        self,
        prompt_text="The capital of France is",
        max_new_tokens=32,
        return_logprob=False,
        top_logprobs_num=0,
        n=1,
    ):
        input_ids = _tokenizer(prompt_text, return_tensors="pt")["input_ids"][
            0
        ].tolist()

        response = requests.post(
            _base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": max_new_tokens,
                    "n": n,
                    "stop_token_ids": [_tokenizer.eos_token_id],
                },
                "stream": False,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "logprob_start_len": 0,
            },
        )
        ret = response.json()
        print(json.dumps(ret, indent=2))

        def assert_one_item(item):
            if item["meta_info"]["finish_reason"]["type"] == "stop":
                self.assertEqual(
                    item["meta_info"]["finish_reason"]["matched"],
                    _tokenizer.eos_token_id,
                )
            elif item["meta_info"]["finish_reason"]["type"] == "length":
                self.assertEqual(
                    len(item["token_ids"]), item["meta_info"]["completion_tokens"]
                )
                self.assertEqual(len(item["token_ids"]), max_new_tokens)
                self.assertEqual(item["meta_info"]["prompt_tokens"], len(input_ids))

                if return_logprob:
                    self.assertEqual(
                        len(item["meta_info"]["input_token_logprobs"]),
                        len(input_ids),
                        f'{len(item["meta_info"]["input_token_logprobs"])} mismatch with {len(input_ids)}',
                    )
                    self.assertEqual(
                        len(item["meta_info"]["output_token_logprobs"]),
                        max_new_tokens,
                    )

        # Determine whether to assert a single item or multiple items based on n
        if n == 1:
            assert_one_item(ret)
        else:
            self.assertEqual(len(ret), n)
            for i in range(n):
                assert_one_item(ret[i])

        print("=" * 100)

    def test_simple_decode(self):
        self.run_decode()

    def test_parallel_sample(self):
        self.run_decode(n=3)

    def test_logprob(self):
        for top_logprobs_num in [0, 3]:
            self.run_decode(return_logprob=True, top_logprobs_num=top_logprobs_num)

    def test_eos_behavior(self):
        self.run_decode(max_new_tokens=256)


if __name__ == "__main__":
    unittest.main()
