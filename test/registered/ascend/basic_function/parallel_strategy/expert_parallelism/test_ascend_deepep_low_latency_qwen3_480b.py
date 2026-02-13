import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_WEIGHTS_PATH,
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

register_npu_ci(est_time=200, suite="nightly-16-npu-a3", nightly=True)


class TestDeepEpQwen(CustomTestCase):
    """
    Testcase:Test the Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot model with DeepEP's low_latency mode enabled,
    and verify that there is no drop in accuracy compared to when DeepEP is not enabled.

    [Test Category] Parameter
    [Test Target] --moe-a2a-backend, --deepep-mode
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--nnodes",
                "1",
                "--node-rank",
                "0",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--quantization",
                "modelslim",
                "--max-running-requests",
                96,
                "--context-length",
                8192,
                "--dtype",
                "bfloat16",
                "--chunked-prefill-size",
                1024,
                "--max-prefill-tokens",
                458880,
                "--disable-radix-cache",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--tp-size",
                16,
                "--dp-size",
                4,
                "--enable-dp-attention",
                "--enable-dp-lm-head",
                "--mem-fraction-static",
                0.7,
                "--cuda-graph-bs",
                16,
                20,
                24,
            ],
            env={
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
                "HCCL_BUFFSIZE": "2100",
                "HCCL_OP_EXPANSION_MODE": "AIV",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        expect_score = 0.61
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
        expect_accuracy = 0.91
        args = SimpleNamespace(
            num_shots=8,
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
