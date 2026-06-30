import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestTokenizerBatchEncode(CustomTestCase):
    """Testcase: Verify throughput improvement when enabling --enable-tokenizer-batch-encode.

    [Test Category] Parameter
    [Test Target] --enable-tokenizer-batch-encode
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--attention-backend",
            "ascend",
            "--enable-tokenizer-batch-encode",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )
        return response.json()

    def test_tokenizer_batch_encode_throughput_improvement(self):
        self.run_decode(16)
        max_tokens = 256
        tic = time.perf_counter()
        res = self.run_decode(max_tokens)
        tok = time.perf_counter()
        print(f"{res=}")
        throughput = max_tokens / (tok - tic)
        self.assertGreaterEqual(throughput, 200)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.38)


if __name__ == "__main__":
    unittest.main()
