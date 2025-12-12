import json
import random
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import numpy as np
import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k_eval
from sglang.test.server_fixtures.eagle_fixture import EagleServerBase
from sglang.test.test_utils import (
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    run_logprob_check,
)

register_cuda_ci(est_time=473, suite="stage-b-test-small-1-gpu")


class TestEAGLEServerBasic(EagleServerBase):
    extra_args = ["--chunked-prefill-size", 128, "--max-running-requests", 8]

    # FIXME(lsyin): move the test methods to kits
    def test_request_abort(self):
        concurrency = 4
        threads = [
            threading.Thread(target=self.send_request) for _ in range(concurrency)
        ] + [
            threading.Thread(target=self.send_requests_abort)
            for _ in range(concurrency)
        ]
        for worker in threads:
            worker.start()
        for p in threads:
            p.join()

    def test_max_token_one(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=1,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        # Just run and check it does not hang
        metrics = run_gsm8k_eval(args)
        self.assertGreater(metrics["output_throughput"], 50)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        metrics = run_gsm8k_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.20)

        server_info = requests.get(self.base_url + "/get_server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        speculative_eagle_topk = server_info["speculative_eagle_topk"]

        if speculative_eagle_topk == 1:
            self.assertGreater(avg_spec_accept_length, 2.5)
        else:
            self.assertGreater(avg_spec_accept_length, 3.5)

        # Wait a little bit so that the memory check happens.
        time.sleep(4)

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
                "logprob_start_len": logprob_start_len,
            },
        )
        response_json = response.json()
        print(json.dumps(response_json, indent=2))

        for res in response_json:
            self.assertEqual(
                res["meta_info"]["prompt_tokens"],
                logprob_start_len + len(res["meta_info"]["input_token_logprobs"]),
            )

            self.assertEqual(res["meta_info"]["completion_tokens"], new_tokens)
            self.assertEqual(len(res["meta_info"]["output_token_logprobs"]), new_tokens)

    def test_logprob_match(self):
        """Test the output logprobs are close to the input logprobs if we run a prefill again."""

        def run_generate(
            prompt,
            return_logprob=False,
            max_new_tokens=512,
            logprob_start_len=-1,
            temperature=1.0,
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
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": return_logprob,
                    "return_text_in_logprobs": True,
                    "logprob_start_len": logprob_start_len,
                    "temp_scaled_logprobs": True,
                },
            )
            return response.json()

        prompt = "I have a very good idea on how to"

        for temperature in [1.0]:
            gen = run_generate(
                prompt,
                return_logprob=True,
                logprob_start_len=0,
                temperature=temperature,
            )
            output_logprobs = np.array(
                [x[0] for x in gen["meta_info"]["output_token_logprobs"]]
            )
            num_prompts_tokens = gen["meta_info"]["prompt_tokens"]

            input_tokens = [x[1] for x in gen["meta_info"]["input_token_logprobs"]]
            output_tokens = [x[1] for x in gen["meta_info"]["output_token_logprobs"]]

            new_prompt = input_tokens + output_tokens
            score = run_generate(
                new_prompt,
                return_logprob=True,
                logprob_start_len=0,
                max_new_tokens=0,
                temperature=temperature,
            )
            output_logprobs_score = np.array(
                [
                    x[0]
                    for x in score["meta_info"]["input_token_logprobs"][
                        num_prompts_tokens:
                    ]
                ]
            )

            print(f"{output_logprobs[-10:]=}")
            print(f"{output_logprobs_score[-10:]=}")

            diff = np.abs(output_logprobs - output_logprobs_score)
            max_diff = np.max(diff)
            self.assertLess(max_diff, 0.255)

    def test_logprob_mixed(self):
        args = []
        temperature = 0
        # input_len, output_len, temperature, logprob_start_len, return_logprob, top_logprobs_num
        # Llama 2 context length seems to be only 2k, so we can only test small length.
        for input_len in [200, 500, 1000, 2000]:
            for output_len in [4, 8]:
                for logprob_start_len in [0, 100, 300, 800, 1998]:
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

    def test_constrained_decoding(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me a json"},
        ]

        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        self.assertEqual(response.status_code, 200)
        res = response.json()

        # Validate response structure
        self.assertIn("choices", res)
        self.assertEqual(len(res["choices"]), 1)
        self.assertIn("message", res["choices"][0])
        self.assertIn("content", res["choices"][0]["message"])

        # Validate JSON content
        content_json = res["choices"][0]["message"]["content"]
        is_valid_json = True
        try:
            content = json.loads(content_json)
            self.assertIsInstance(content, dict)
        except Exception:
            print(f"parse JSON failed: {content_json}")
            is_valid_json = False
        self.assertTrue(is_valid_json)


class TestEAGLERetract(TestEAGLEServerBasic):
    extra_args = ["--chunked-prefill-size", 128, "--max-running-requests", 64]

    @classmethod
    def setUpClass(cls):
        # These config helps find a leak.
        # FIXME(lsyin): use override context manager
        envs.SGLANG_CI_SMALL_KV_SIZE.set(4500)
        super().setUpClass()


class TestEAGLEServerTriton(TestEAGLEServerBasic):
    extra_args = ["--attention-backend=triton", "--max-running-requests=8"]


class TestEAGLEServerPageSize(TestEAGLEServerBasic):
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    extra_args = [
        "--chunked-prefill-size=128",
        "--max-running-requests=8",
        "--page-size=4",
        "--attention-backend=flashinfer",
    ]


class TestEAGLEServerPageSizeTopk(TestEAGLEServerBasic):
    # default topk=8 and tokens=64
    extra_args = [
        "--chunked-prefill-size=128",
        "--max-running-requests=8",
        "--page-size=4",
        "--attention-backend=flashinfer",
    ]


if __name__ == "__main__":
    unittest.main()
