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


class TestPiecewiseCudaGraphBenchmark(CustomTestCase):

    def test_latency(self):
        prefill_latency, _, _ = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST,
            other_args=["--enable-piecewise-cuda-graph"],
        )
        self.assertLess(prefill_latency, 0.015)


if __name__ == "__main__":
    unittest.main()
