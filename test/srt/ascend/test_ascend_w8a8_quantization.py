"""
Usage:
python3 -m unittest test_ascend_w8a8_quantization.TestAscendW8A8.test_gsm8k
"""

import os
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
DEFAULT_PORT_FOR_SRT_TEST_RUNNER = (
    7000 + int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0]) * 100
)
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000}"


class TestAscendW8A8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "vllm-ascend/Qwen2.5-0.5B-Instruct-w8a8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--disable-cuda-graph",
                "--device",
                "npu",
                "--attention-backend",
                "ascend",
                "--quantization",
                "w8a8_int8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        base_url = DEFAULT_URL_FOR_TEST
        url = urlparse(base_url)
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{url.hostname}",
            port=int(url.port),
        )
        metrics = run_eval(args)
        print(metrics)

        self.assertGreaterEqual(metrics["accuracy"], 0.25)
        self.assertGreaterEqual(metrics["output_throughput"], 1000)

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "ignore_eos": True,
            },
        )
        return response.json()

    def test_throughput(self):
        max_tokens = 256

        tic = time.perf_counter()
        res = self.run_decode(max_tokens)
        tok = time.perf_counter()
        print(res["text"])
        throughput = max_tokens / (tok - tic)
        print(f"Throughput: {throughput} tokens/s")

        if is_in_ci():
            self.assertGreaterEqual(throughput, 25)


if __name__ == "__main__":
    unittest.main()
