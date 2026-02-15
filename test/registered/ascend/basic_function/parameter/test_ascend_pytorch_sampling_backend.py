import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestPyTorchSamplingBackend(CustomTestCase):
    """Testcase：When --sampling-backend=pytorch, verify the MMLU dataset accuracy (>0.65) and greedy sampling consistency of the Llama-3.1-8B-Instruct mode

    [Test Category] Parameter
    [Test Target] --sampling-backend;
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--sampling-backend",
                "pytorch",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        """Verify MMLU dataset evaluation accuracy meets the minimum requirement (score ≥ 0.65)"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)

    def test_greedy(self):
        """Verify greedy sampling consistency (identical results for single/batch requests with temperature=0)"""

        first_text = None

        # 1. Verify consistency of results for 5 consecutive single requests
        for _ in range(5):
            response_single = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of Germany is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            ).json()
            text = response_single["text"]
            if first_text is None:
                first_text = text

            self.assertEqual(text, first_text)

        first_text = None

        # 2. Send a batch request with 10 identical prompts
        response_batch = requests.post(
            self.base_url + "/generate",
            json={
                "text": ["The capital of Germany is"] * 10,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        ).json()

        # 3. Verify consistency of results within the batch response
        for i in range(10):
            text = response_batch[i]["text"]
            if first_text is None:
                first_text = text
            self.assertEqual(text, first_text)


if __name__ == "__main__":
    unittest.main()
