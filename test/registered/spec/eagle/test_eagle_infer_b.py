import json
import random
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k_eval
from sglang.test.kits.abort_timeout_kit import (
    AbortAllMixin,
    RunningTimeoutTwoWaveMixin,
    WaitingTimeoutMixin,
)
from sglang.test.kits.logprob_kit import LogprobCrossModeMixin
from sglang.test.kits.radix_cache_server_kit import run_radix_attention_test
from sglang.test.server_fixtures.eagle_fixture import EagleServerBase
from sglang.test.test_utils import DEFAULT_TARGET_MODEL_EAGLE

register_cuda_ci(est_time=1100, suite="stage-b-test-large-1-gpu")


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

    def test_radix_attention(self):
        run_radix_attention_test(self.base_url)
        self.assertIsNone(self.process.poll())

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

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        speculative_eagle_topk = server_info["speculative_eagle_topk"]

        if speculative_eagle_topk == 1:
            self.assertGreater(avg_spec_accept_length, 2.5)
        else:
            self.assertGreater(avg_spec_accept_length, 3.49)

        # Wait a little bit so that the memory check happens.
        time.sleep(4)

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
                "model": DEFAULT_TARGET_MODEL_EAGLE,
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
    extra_args = [
        "--chunked-prefill-size=128",
        "--max-running-requests=64",
        "--max-total-tokens=4500",  # Set a smaller KV cache to trigger retract more easily
    ]

    @classmethod
    def setUpClass(cls):
        # These config helps find a leak.
        with envs.SGLANG_TEST_RETRACT.override(True):
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

    @classmethod
    def setUpClass(cls):
        # Runtime check only supported for topk=1, and can help to find a leak.
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1):
            super().setUpClass()


class TestEAGLEServerPageSizeTopk(TestEAGLEServerBasic):
    # default topk=8 and tokens=64
    extra_args = [
        "--chunked-prefill-size=128",
        "--max-running-requests=8",
        "--page-size=4",
        "--attention-backend=flashinfer",
    ]


class TestEAGLEServerPageSizeTopkFA3(TestEAGLEServerBasic):
    # default topk=8 and tokens=64
    spec_topk = 5
    spec_steps = 8
    spec_tokens = 64

    extra_args = [
        "--page-size=256",
        "--attention-backend=fa3",
        "--cuda-graph-max-bs=5",
        "--dtype=float16",
        "--max-running-requests=8",
    ]


class TestEAGLEAbortAll(AbortAllMixin, EagleServerBase):
    abort_all_max_new_tokens = 4000
    extra_args = ["--max-running-requests=8"]


class TestEAGLEWaitingTimeout(WaitingTimeoutMixin, EagleServerBase):
    extra_args = ["--max-running-requests=1"]

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(0.001):
            super().setUpClass()


class TestEAGLERunningTimeout(RunningTimeoutTwoWaveMixin, EagleServerBase):
    # Regression test for https://github.com/sgl-project/sglang/pull/18760
    extra_args = ["--max-running-requests=16"]

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(3):
            super().setUpClass()


class TestEAGLELogprobCrossMode(EagleServerBase, LogprobCrossModeMixin):
    logprob_decimal_places = 1
    extra_args = ["--chunked-prefill-size", 128, "--max-running-requests", 8]


if __name__ == "__main__":
    unittest.main()
