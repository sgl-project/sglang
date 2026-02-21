import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_235B_A22B_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestDeepEpAutoQwen3235B(CustomTestCase):
    """Testcase: This test case verifies that the Qwen3-235B-A22B-W8A8 model with DeepEP's auto mode achieves an
    accuracy of greater than or equal to 0.5 on MMLU and greater than or equal to 0.94 on GSM8K.

    [Test Category] Parameter
    [Test Target] --moe-a2a-backend deepep;--deepep-mode auto
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_235B_A22B_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--tp-size",
                "8",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--disable-cuda-graph",
                "--dp-size",
                2,
                "--enable-dp-attention",
                "--chunked-prefill-size",
                1024,
                "--quantization",
                "modelslim",
                "--mem-fraction-static",
                "0.75",
            ],
            env={
                "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
                "HCCL_BUFFSIZE": "2048",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=8,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.5)

    def test_gsm8k(self):
        expect_accuracy = 0.94
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_gsm8k(args)
        achieved_accuracy = metrics["accuracy"]
        self.assertGreaterEqual(
            achieved_accuracy,
            expect_accuracy,
            f"Accuracy of {self.model} is {str(achieved_accuracy)}, is lower than {expect_accuracy}",
        )
        print(f"Model {self.model} achieved accuracy: {str(achieved_accuracy)}")


if __name__ == "__main__":
    unittest.main()
