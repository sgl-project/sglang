import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=126, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=1100, suite="stage-b-test-1-gpu-small-amd")


class TestTorchCompile(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-torch-compile", "--cuda-graph-max-bs", "4"],
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

    def test_throughput(self):
        # Warmup
        res = self.run_decode(16)

        max_tokens = 256
        tic = time.perf_counter()
        res = self.run_decode(max_tokens)
        tok = time.perf_counter()
        print(f"{res=}")
        throughput = max_tokens / (tok - tic)
        print(f"Throughput: {throughput} tokens/s")

        if is_in_amd_ci():
            self.assertGreaterEqual(throughput, 145)
        else:
            self.assertGreaterEqual(throughput, 152)


if __name__ == "__main__":
    unittest.main()
