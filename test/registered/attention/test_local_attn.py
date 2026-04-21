import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_LOCAL_ATTENTION,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Local attention with FA3 (requires SM 90+ / H100, tp=4)
register_cuda_ci(est_time=208, suite="stage-c-test-4-gpu-h100")


@unittest.skipIf(get_device_sm() < 90, "Test requires CUDA SM 90 or higher")
class TestFlashAttention3LocalAttn(CustomTestCase):
    model = DEFAULT_MODEL_NAME_FOR_TEST_LOCAL_ATTENTION
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.90

    @classmethod
    def get_server_args(cls):
        return [
            "--cuda-graph-max-bs",
            "2",
            "--attention-backend",
            "fa3",
            "--tp",
            "4",
            "--context-length",
            "1000000",
        ]

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
            env=os.environ,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=128,
            num_shots=4,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        # Use the appropriate metric key based on the test class
        metric_key = "score"
        self.assertGreater(metrics[metric_key], self.accuracy_threshold)


if __name__ == "__main__":
    unittest.main()
