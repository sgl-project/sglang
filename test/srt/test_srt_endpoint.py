"""
python3 -m unittest test_srt_endpoint.TestSRTEndpoint.test_simple_decode
python3 -m unittest test_srt_endpoint.TestSRTEndpoint.test_logprob_with_chunked_prefill
"""

import json
import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
import requests

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_logprob_check,
)


class TestSRTEndpoint(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--enable-custom-logit-processor",
                "--mem-fraction-static",
                "0.7",
                "--cuda-graph-max-bs",
                "8",
            ),
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
                logprob_start_len + len(res["meta_info"]["input_token_logprobs"]),
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
                "top_logprobs_num": 5,
            },
        )
        response_json = response.json()
        # print(json.dumps(response_json, indent=2))

        res = response_json
        self.assertEqual(res["meta_info"]["completion_tokens"], new_tokens)

        # Test the number of tokens are correct
        self.assertEqual(len(res["meta_info"]["output_token_logprobs"]), new_tokens)
        self.assertEqual(len(res["meta_info"]["output_top_logprobs"]), new_tokens)

        # Test the top-1 tokens are the same as output tokens (because temp = 0.0)
        for i in range(new_tokens):
            self.assertListEqual(
                res["meta_info"]["output_token_logprobs"][i],
                res["meta_info"]["output_top_logprobs"][i][0],
            )
            self.assertEqual(len(res["meta_info"]["output_top_logprobs"][i]), 5)

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
        self.assertLess(max_diff, 0.35)

    def test_logprob_mixed(self):
        args = []
        temperature = 0
        # input_len, output_len, temperature, logprob_start_len, return_logprob, top_logprobs_num
        for input_len in [1000, 5000, 10000, 50000]:
            for output_len in [4, 8]:
                for logprob_start_len in [0, 500, 2500, 5000, 25000]:
                    for return_logprob in [True, False]:
                        for top_logprobs_num in [0, 5]:

                            if logprob_start_len >= input_len:
                                continue

                            args.append(
                                (
                                    input_len,
                                    output_len,
                                    temperature,
                                    logprob_start_len,
                                    return_logprob,
                                    top_logprobs_num,
                                )
                            )

        random.shuffle(args)

        func = partial(run_logprob_check, self)
        with ThreadPoolExecutor(8) as executor:
            list(executor.map(func, args))

    def test_logprob_grammar(self):
        prompts = "Question: Is Paris the Capital of France? Answer:"
        allowed_tokens = [" Yes", " No"]

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 1.0,
                    "max_new_tokens": 1,
                    "regex": "( Yes| No)",
                },
                "return_logprob": True,
                "top_logprobs_num": 5,  # The grammar constraint allows all prefix tokens so we need to use a larger top_k.
                "return_text_in_logprobs": True,
            },
        )
        response_json = response.json()
        output_top_logprobs = response_json["meta_info"]["output_top_logprobs"][0]
        print(f"{output_top_logprobs=}")

        # Parse results
        # This is becaues the grammar constraint allows all prefix tokens
        logprobs = [None] * 2
        for i in range(len(output_top_logprobs)):
            try:
                idx = allowed_tokens.index(output_top_logprobs[i][2])
            except ValueError:
                # Not found
                continue
            logprobs[idx] = output_top_logprobs[i][0]

        self.assertTrue(all(x is not None for x in logprobs))

    def run_custom_logit_processor(self, target_token_id: Optional[int] = None):
        """Test custom logit processor with custom params.

        If target_token_id is None, the custom logit processor won't be passed in.
        """

        custom_params = {"token_id": target_token_id}

        class DeterministicLogitProcessor(CustomLogitProcessor):
            """A dummy logit processor that changes the logits to always
            sample the given token id.
            """

            def __call__(self, logits, custom_param_list):
                assert logits.shape[0] == len(custom_param_list)
                key = "token_id"

                for i, param_dict in enumerate(custom_param_list):
                    # Mask all other tokens
                    logits[i, :] = -float("inf")
                    # Assign highest probability to the specified token
                    logits[i, param_dict[key]] = 0.0
                return logits

        prompts = "Question: Is Paris the Capital of France? Answer:"

        # Base case json data to be posted to the server.
        base_json = {
            "text": prompts,
            "sampling_params": {"temperature": 0.0},
            "return_logprob": True,
        }

        # Custom json data with custom logit processor and params.
        custom_json = base_json.copy()
        # Only set the custom logit processor if target_token_id is not None.
        if target_token_id is not None:
            custom_json["custom_logit_processor"] = (
                DeterministicLogitProcessor().to_str()
            )
            custom_json["sampling_params"]["custom_params"] = custom_params

        custom_response = requests.post(
            self.base_url + "/generate",
            json=custom_json,
        ).json()

        output_token_logprobs = custom_response["meta_info"]["output_token_logprobs"]
        sampled_tokens = [x[1] for x in output_token_logprobs]

        # The logit processor should always sample the given token as the logits is deterministic.
        if target_token_id is not None:
            self.assertTrue(
                all(x == custom_params["token_id"] for x in sampled_tokens),
                # Print the detailed test case info if the test fails.
                f"{target_token_id=}\n{sampled_tokens=}\n{custom_response=}",
            )

    def run_stateful_custom_logit_processor(
        self, first_token_id: int | None, delay: int = 2
    ):
        """Test custom logit processor with custom params and state.

        Should sample the first `delay` tokens normally, then output first_token_id and consecutive tokens after that.
        If first_token_id is None, the custom logit processor won't be passed in.
        """

        custom_params = {"token_id": first_token_id, "delay": 2}

        class DeterministicStatefulLogitProcessor(CustomLogitProcessor):
            """A dummy logit processor that changes the logits to always
            sample the given token id.
            """

            def __call__(self, logits, custom_param_list):
                assert logits.shape[0] == len(custom_param_list)

                for i, param_dict in enumerate(custom_param_list):
                    if param_dict["delay"] > 0:
                        param_dict["delay"] -= 1
                        continue
                    if param_dict["delay"] == 0:
                        param_dict["delay"] -= 1
                        force_token = param_dict["token_id"]
                    else:
                        output_ids = param_dict["__req__"].output_ids
                        force_token = output_ids[-1] + 1
                    # Mask all other tokens
                    logits[i, :] = -float("inf")
                    # Assign highest probability to the specified token
                    logits[i, force_token] = 0.0
                return logits

        prompts = "Question: Is Paris the Capital of France? Answer:"

        # Base case json data to be posted to the server.
        base_json = {
            "text": prompts,
            "sampling_params": {"temperature": 0.0},
            "return_logprob": True,
        }

        # Custom json data with custom logit processor and params.
        custom_json = base_json.copy()
        # Only set the custom logit processor if target_token_id is not None.
        if first_token_id is not None:
            custom_json["custom_logit_processor"] = (
                DeterministicStatefulLogitProcessor().to_str()
            )
            custom_json["sampling_params"]["custom_params"] = custom_params

        custom_response = requests.post(
            self.base_url + "/generate",
            json=custom_json,
        ).json()

        output_token_logprobs = custom_response["meta_info"]["output_token_logprobs"]
        sampled_tokens = [x[1] for x in output_token_logprobs]
        # The logit processor should always sample the given token as the logits is deterministic.
        if first_token_id is not None:
            self.assertTrue(
                all(
                    x == custom_params["token_id"] + k
                    for k, x in enumerate(sampled_tokens[custom_params["delay"] :])
                ),
                # Print the detailed test case info if the test fails.
                f"{first_token_id=}\n{sampled_tokens=}\n{custom_response=}",
            )

    def test_custom_logit_processor(self):
        """Test custom logit processor with a single request."""
        self.run_custom_logit_processor(target_token_id=5)

    def test_custom_logit_processor_batch_mixed(self):
        """Test a batch of requests mixed of requests with and without custom logit processor."""
        target_token_ids = list(range(32)) + [None] * 16
        random.shuffle(target_token_ids)
        with ThreadPoolExecutor(len(target_token_ids)) as executor:
            list(executor.map(self.run_custom_logit_processor, target_token_ids))

    def test_stateful_custom_logit_processor(self):
        """Test custom logit processor with a single request."""
        self.run_stateful_custom_logit_processor(first_token_id=5)

    def test_stateful_custom_logit_processor_batch_mixed(self):
        """Test a batch of requests mixed of requests with and without custom logit processor."""
        target_token_ids = list(range(32)) + [None] * 16
        random.shuffle(target_token_ids)
        with ThreadPoolExecutor(len(target_token_ids)) as executor:
            list(
                executor.map(self.run_stateful_custom_logit_processor, target_token_ids)
            )

    def test_cache_tokens(self):
        for _ in range(2):
            time.sleep(1)
            response = requests.post(self.base_url + "/flush_cache")
            assert response.status_code == 200

        def send_and_check_cached_tokens(input_ids):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": list(input_ids),
                    "sampling_params": {
                        "max_new_tokens": 1,
                    },
                },
            )
            response_json = response.json()
            return response_json["meta_info"]["cached_tokens"]

        self.assertEqual(send_and_check_cached_tokens(range(0, 100)), 0)
        self.assertEqual(send_and_check_cached_tokens(range(0, 10000)), 100)
        self.assertEqual(send_and_check_cached_tokens(range(0, 10000)), 9999)
        self.assertEqual(send_and_check_cached_tokens(range(0, 1000)), 999)
        self.assertEqual(send_and_check_cached_tokens(range(0, 11000)), 10000)

    def test_get_server_info(self):
        response = requests.get(self.base_url + "/get_server_info")
        response_json = response.json()

        max_total_num_tokens = response_json["max_total_num_tokens"]
        self.assertIsInstance(max_total_num_tokens, int)

        attention_backend = response_json["attention_backend"]
        self.assertIsInstance(attention_backend, str)

        version = response_json["version"]
        self.assertIsInstance(version, str)

    def test_get_server_info_concurrent(self):
        """Make sure the concurrent get_server_info doesn't crash the server."""
        tp = ThreadPoolExecutor(max_workers=30)

        def s():
            server_info = requests.get(self.base_url + "/get_server_info")
            server_info.json()

        futures = []
        for _ in range(4):
            futures.append(tp.submit(s))

        for f in futures:
            f.result()


if __name__ == "__main__":
    unittest.main()
