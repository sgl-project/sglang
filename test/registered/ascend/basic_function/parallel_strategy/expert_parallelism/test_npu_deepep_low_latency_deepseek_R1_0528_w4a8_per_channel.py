import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-16-npu-a3", nightly=True)


class TestDeepepLowlatencyDeepseekR1(CustomTestCase):
    """Testcase: Verify the accuracy of DeepSeek-R1 model on MMLU and GSM8K tasks with --deepep-mode low_latency on Ascend NPU backend.

    [Test Category] Parameter
    [Test Target] --speculative-algorithm; --deepep-mode
    """

    accuracy = 0.96

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "16",
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--quantization",
                "modelslim",
                "--watchdog-timeout",
                "9000",
                "--cuda-graph-bs",
                "8",
                "16",
                "24",
                "28",
                "32",
                "36",
                "--mem-fraction-static",
                "0.6",
                "--max-running-requests",
                "144",
                "--context-length",
                "8188",
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "512",
                "--max-prefill-tokens",
                "4096",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--enable-dp-attention",
                "--dp-size",
                "4",
                "--enable-dp-lm-head",
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--dtype",
                "bfloat16",
            ],
            env={
                "SGLANG_SET_CPU_AFFINITY": "1",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "STREAMS_PER_DEVICE": "32",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
                "HCCL_BUFFSIZE": "3000",
                "DEEP_NORMAL_MODE_USE_INT8_QUANT": "0",
                "SGLANG_NPU_USE_MLAPO": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "TRANSFORMERS_VERBOSITY": "error",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        # Terminate the model server process and its child processes after all tests in the class complete
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        # Test Scenario: Verify the model's general knowledge performance on the MMLU dataset
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=8,
            num_threads=32,
        )

        # Execute MMLU evaluation and retrieve performance metrics
        metrics = run_eval(args)
        # Assertion: Ensure the MMLU score meets the basic performance threshold
        self.assertGreater(metrics["score"], 0.5)

    def test_gsm8k(self):
        # Test Scenario: Verify the model's accuracy on GSM8K dataset (mathematical reasoning evaluation)
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            data_path=None,
            num_examples=200,
            num_threads=300,
            num_shots=5,
            max_tokens=512,
        )
        # Execute GSM8K evaluation and get metrics
        metrics = run_eval(args)
        # Assertion: The GSM8K accuracy is not lower than the preset threshold (0.96)
        self.assertGreaterEqual(
            metrics["score"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["score"])}, is lower than {self.accuracy}',
        )


if __name__ == "__main__":
    unittest.main()
