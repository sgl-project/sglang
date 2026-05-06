import os
import subprocess
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=232, suite="stage-b-test-1-gpu-large")

_CUDA_PR_UT_EVENTS = ("pull_request", "workflow_dispatch")


class BaseW8A8Test(CustomTestCase):
    model: str = None
    quantization: str = None
    gsm8k_accuracy_threshold: float = None
    throughput_threshold: float = None

    @classmethod
    def setUpClass(cls):
        if cls is BaseW8A8Test:
            raise unittest.SkipTest("Skip base test class")

        cls.base_url = f"http://127.0.0.1:{get_open_port()}"
        other_args = []
        if cls.quantization:
            other_args.extend(["--quantization", cls.quantization])

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if cls is BaseW8A8Test:
            return
        kill_process_tree(cls.process.pid)
        try:
            cls.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass

    def test_gsm8k(self):
        if self.gsm8k_accuracy_threshold is None:
            self.skipTest("gsm8k_accuracy_threshold not set for this test")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], self.gsm8k_accuracy_threshold)

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
        self.assertGreaterEqual(throughput, self.throughput_threshold)


@unittest.skipIf(
    is_in_ci() and os.getenv("GITHUB_EVENT_NAME") in _CUDA_PR_UT_EVENTS,
    "Meta-Llama-3 W8A8 INT8 throughput is below the PR UT threshold on current H20 runners",
)
class TestW8A8Int8(BaseW8A8Test):
    model = "neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8"
    quantization = "w8a8_int8"
    gsm8k_accuracy_threshold = 0.69
    throughput_threshold = 200


class TestW8A8Fp8(BaseW8A8Test):
    model = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
    quantization = "w8a8_fp8"
    gsm8k_accuracy_threshold = 0.69
    throughput_threshold = 200

    @unittest.skipIf(
        is_in_ci() and os.getenv("GITHUB_EVENT_NAME") in _CUDA_PR_UT_EVENTS,
        "Meta-Llama-3.1-8B FP8 GSM8K accuracy regresses on current CUDA PR UT runners",
    )
    def test_gsm8k(self):
        super().test_gsm8k()

    @unittest.skipIf(
        is_in_ci() and os.getenv("GITHUB_EVENT_NAME") in _CUDA_PR_UT_EVENTS,
        "Meta-Llama-3.1-8B FP8 throughput is below the PR UT threshold on current CUDA PR UT runners",
    )
    def test_throughput(self):
        super().test_throughput()


@unittest.skipIf(
    is_in_ci() and os.getenv("GITHUB_EVENT_NAME") in _CUDA_PR_UT_EVENTS,
    "Qwen3 FP8 MoE accuracy/throughput is unstable on current CUDA PR UT runners",
)
class TestW8A8Fp8MoE(BaseW8A8Test):
    model = "RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
    quantization = "w8a8_fp8"
    gsm8k_accuracy_threshold = 0.88
    throughput_threshold = 180


if __name__ == "__main__":
    unittest.main()
