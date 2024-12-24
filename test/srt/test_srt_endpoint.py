"""
python3 -m unittest test_srt_endpoint.TestSRTEndpoint.test_simple_decode
python3 -m unittest test_srt_endpoint.TestSRTEndpoint.test_logprob_with_chunked_prefill
"""

import json
import unittest

import numpy as np
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestSRTEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(
        self,
        return_logprob=False,
        top_logprobs_num=0,
        return_text=False,
        n=1,
        stream=False,
        batch=False,
    ):
        if batch:
            text = ["The capital of France is"]
        else:
            text = "The capital of France is"

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": text,
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

    def test_simple_decode_batch(self):
        self.run_decode(batch=True)

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
            self.assertEqual(
                res["meta_info"]["prompt_tokens"],
                logprob_start_len + 1 + len(res["meta_info"]["input_token_logprobs"]),
            )
            assert prompts[i].endswith(
                "".join([x[-1] for x in res["meta_info"]["input_token_logprobs"]])
            )

            self.assertEqual(res["meta_info"]["completion_tokens"], new_tokens)
            self.assertEqual(len(res["meta_info"]["output_token_logprobs"]), new_tokens)
            self.assertEqual(
                res["text"],
                "".join([x[-1] for x in res["meta_info"]["output_token_logprobs"]]),
            )

    def test_logprob_with_chunked_prefill(self):
        """Test a long prompt that requests output logprobs will not hit OOM."""
        new_tokens = 4
        prompts = "I have a very good idea on this. " * 8000

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": new_tokens,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
            },
        )
        response_json = response.json()
        print(json.dumps(response_json, indent=2))

        res = response_json
        self.assertEqual(res["meta_info"]["completion_tokens"], new_tokens)
        self.assertEqual(len(res["meta_info"]["output_token_logprobs"]), new_tokens)

    def test_logprob_match(self):
        """Test the output logprobs are close to the input logprobs if we run a prefill again."""

        def run_generate(
            prompt, return_logprob=False, max_new_tokens=512, logprob_start_len=-1
        ):

            if isinstance(prompt, str):
                prompt_kwargs = {"text": prompt}
            else:
                prompt_kwargs = {"input_ids": prompt}

            response = requests.post(
                self.base_url + "/generate",
                json={
                    **prompt_kwargs,
                    "sampling_params": {
                        "temperature": 1.0,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": return_logprob,
                    "return_text_in_logprobs": True,
                    "logprob_start_len": logprob_start_len,
                },
            )
            return response.json()

        prompt = "I have a very good idea on how to"

        gen = run_generate(prompt, return_logprob=True, logprob_start_len=0)
        output_logprobs = np.array(
            [x[0] for x in gen["meta_info"]["output_token_logprobs"]]
        )
        num_prompts_tokens = gen["meta_info"]["prompt_tokens"]

        input_tokens = [x[1] for x in gen["meta_info"]["input_token_logprobs"]]
        output_tokens = [x[1] for x in gen["meta_info"]["output_token_logprobs"]]

        new_prompt = input_tokens + output_tokens
        score = run_generate(
            new_prompt, return_logprob=True, logprob_start_len=0, max_new_tokens=0
        )
        output_logprobs_score = np.array(
            [
                x[0]
                for x in score["meta_info"]["input_token_logprobs"][num_prompts_tokens:]
            ]
        )

        print(f"{output_logprobs[-10:]=}")
        print(f"{output_logprobs_score[-10:]=}")

        diff = np.abs(output_logprobs - output_logprobs_score)
        max_diff = np.max(diff)
        self.assertLess(max_diff, 0.25)

    def test_get_server_info(self):
        response = requests.get(self.base_url + "/get_server_info")
        response_json = response.json()

        max_total_num_tokens = response_json["max_total_num_tokens"]
        self.assertIsInstance(max_total_num_tokens, int)

        attention_backend = response_json["attention_backend"]
        self.assertIsInstance(attention_backend, str)

        version = response_json["version"]
        self.assertIsInstance(version, str)


if __name__ == "__main__":
    unittest.main()
