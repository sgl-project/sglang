"""
Usage:
python3 -m unittest test_ascend_w4a4_quantization.TestAscendW4A4.test_gsm8k
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


class TestAscendW4A4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/msit/Qwen3-8B-W4A4/"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--device",
                "npu",
                "--attention-backend",
                "ascend",
                "--quantization",
                "w4a4_int4",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.8",
                "--cuda-graph-bs",
                "64",
                "--disable-radix-cache",
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
            num_questions=128,
            max_new_tokens=512,
            parallel=64,
            host=f"http://{url.hostname}",
            port=int(url.port),
        )
        metrics = run_eval(args)
        print(metrics)

        self.assertGreaterEqual(metrics["accuracy"], 0.75)
        self.assertGreaterEqual(metrics["output_throughput"], 700)

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
