import json
import os
import random
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import numpy as np
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_logprob_check,
)


class TestEAGLEServer(CustomTestCase):
    PROMPTS = [
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nToday is a sunny day and I like[/INST]"
        '[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nWhat are the mental triggers in Jeff Walker\'s Product Launch Formula and "Launch" book?[/INST]',
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nSummarize Russell Brunson's Perfect Webinar Script...[/INST]",
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwho are you?[/INST]",
        "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwhere are you from?[/INST]",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def send_request(self):
        time.sleep(random.uniform(0, 2))
        for prompt in self.PROMPTS:
            url = self.base_url + "/generate"
            data = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1024,
                },
            }
            response = requests.post(url, json=data)
            assert response.status_code == 200

    def send_requests_abort(self):
        for prompt in self.PROMPTS:
            try:
                time.sleep(random.uniform(0, 2))
                url = self.base_url + "/generate"
                data = {
                    "model": "base",
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1024,
                    },
                }
                # set timeout = 1s, mock disconnected
                requests.post(url, json=data, timeout=1)
            except Exception as e:
                print(e)
                pass

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
        metrics = run_eval(args)
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

        metrics = run_eval(args)
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

    def run_decode(self, sampling_params):
        return_logprob = True
        top_logprobs_num = 5
        return_text = True
        n = 1

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Human: Write a travel blog post to Hawaii.\n\nAssistant:",
                "sampling_params": {
                    "max_new_tokens": 48,
                    "n": n,
                    "temperature": 0.7,
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


class TestEAGLERetract(TestEAGLEServer):
    @classmethod
    def setUpClass(cls):
        # These config helps find a leak.
        os.environ["SGLANG_CI_SMALL_KV_SIZE"] = "4500"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                64,
            ],
        )


class TestEAGLEServerTriton(TestEAGLEServer):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--attention-backend",
                "triton",
                "--max-running-requests",
                8,
            ],
        )


class TestEAGLEServerPageSize(TestEAGLEServer):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                5,
                "--speculative-eagle-topk",
                1,
                "--speculative-num-draft-tokens",
                6,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
                "--page-size",
                4,
                "--attention-backend",
                "flashinfer",
            ],
        )


class TestEAGLEServerPageSizeTopk(TestEAGLEServer):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
                "--speculative-num-steps",
                5,
                "--speculative-eagle-topk",
                8,
                "--speculative-num-draft-tokens",
                64,
                "--mem-fraction-static",
                0.7,
                "--chunked-prefill-size",
                128,
                "--max-running-requests",
                8,
                "--page-size",
                4,
                "--attention-backend",
                "flashinfer",
            ],
        )


if __name__ == "__main__":
    unittest.main()
