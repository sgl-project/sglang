import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=200,
    suite="nightly-8-npu-a3",
    nightly=True,
)


class TestQwen3Next(CustomTestCase):
    """
    Testcase:Test the Qwen3-Next-80B-A3B-Instruct-W8A8 model with DeepEP's low_latency mode enabled, and verify that
    there is no drop in accuracy compared to when DeepEP is not enabled.

    [Test Category] Parameter
    [Test Target] --moe-a2a-backend deepep, --deepep-mode low_latency
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--tp-size",
                8,
                "--mem-fraction-static",
                0.8,
                "--max-running-requests",
                80,
                "--watchdog-timeout",
                9000,
                "--disable-radix-cache",
                "--cuda-graph-bs",
                2,
                4,
                6,
                8,
                "--chunked-prefill-size",
                1024,
                "--max-prefill-tokens",
                28672,
                "--max-total-tokens",
                450560,
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
            ],
            env={
                # The product of the following two environment variables must be greater than --max-prefill-tokens
                # divide by dp size
                "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3000",
                "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
                # In NPU scenarios, operators only support BF16 precision.
                # This environment variable needs to be set for quantizing weights.
                "SGLANG_DEEPEP_BF16_DISPATCH": "1",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "STREAMS_PER_DEVICE": "32",
                "HCCL_OP_EXPANSION_MODE": "AIV",
                "HCCL_ALGO": "level0:NA;level1:ring",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "160",
                "HCCL_BUFFSIZE": "2048",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        expect_score = 0.56
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=8,
            num_threads=32,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], expect_score)

    def test_gsm8k(self):
        expect_accuracy = 0.9
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
        self.assertGreaterEqual(
            metrics["accuracy"],
            expect_accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {expect_accuracy}',
        )


if __name__ == "__main__":
    unittest.main()
