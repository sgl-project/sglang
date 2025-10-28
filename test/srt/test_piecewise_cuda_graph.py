import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
    run_bench_one_batch,
)


class TestPiecewiseCudaGraphCorrectness(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-piecewise-cuda-graph"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gpqa(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gpqa",
            num_examples=64,
            num_threads=16,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.235)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)


class TestPiecewiseCudaGraphBenchmark(CustomTestCase):

    def test_latency(self):
        prefill_latency, _, _ = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST,
            other_args=["--enable-piecewise-cuda-graph"],
        )
        self.assertLess(prefill_latency, 0.015)


class TestPiecewiseCudaGraphQwen3MoE(CustomTestCase):
    """Test piecewise CUDA graph with Qwen3-Coder-30B-A3B-Instruct MoE model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-cuda-graph",
                "--piecewise-cuda-graph-compiler",
                "eager",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy with 8-shot setting"""
        num_examples = 2000

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=num_examples,
            num_threads=min(num_examples, 1024),
        )

        metrics = run_eval(args)
        print(f"GSM8K Accuracy: {metrics['score']:.3f}")

        self.assertGreaterEqual(metrics["score"], 0.90)


if __name__ == "__main__":
    unittest.main()
