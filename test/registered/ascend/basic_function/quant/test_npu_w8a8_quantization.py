"""
Usage:
python3 -m unittest test_ascend_w8a8_quantization.TestAscendW8A8.test_gsm8k
"""

import os
import time
import unittest

import requests

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    REDHATAI_QWEN2_5_0_5B_INSTRUCT_QUANTIZED_W8A8_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci, write_github_step_summary

register_npu_ci(est_time=400, suite="stage-b-test-1-npu-a2", nightly=False)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestAscendW8A8CompressedTensors(GSM8KAscendMixin, CustomTestCase):
    model = REDHATAI_QWEN2_5_0_5B_INSTRUCT_QUANTIZED_W8A8_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--disable-cuda-graph",
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
    ]
    env = {
        **os.environ,
    }

    # GSM8K Configs
    accuracy = 0.3  # GSM8K accuracy ≥0.3
    num_questions = 200
    gsm8k_num_shots = 5
    output_throughput = 700  # GSM8K output throughput >=700 tokens/s

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
        throughput = max_tokens / (tok - tic)
        summary = f"\nThroughput: {throughput} tokens/s"
        print(res["text"] + summary)

        if is_in_ci():
            write_github_step_summary(summary + "\nThroughput threshold: 25 tokens/s")
            self.assertGreaterEqual(throughput, 25)


if __name__ == "__main__":
    unittest.main()
