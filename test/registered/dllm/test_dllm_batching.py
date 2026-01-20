from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=500, suite="stage-b-test-large-1-gpu")

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

"""
Test dLLM batching capability on CUDA GPUs.

As current dLLM batching performance is suboptimal to BS=1, this test only verifies correctness.
The test will be removed once dLLM batching performance improves.
"""


class TestBatching(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "inclusionAI/LLaDA2.0-mini"
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "4",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "LowConfidence",
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

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.88)


if __name__ == "__main__":
    unittest.main()
