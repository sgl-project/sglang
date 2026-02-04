import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestAscendEagle3(CustomTestCase):
    """Testcase: Verify GSM8K inference accuracy ≥0.81 for model with specified EAGLE3 speculative inference parameters.

    [Test Category] Parameter
    [Test Target] --speculative-draft-model-quantization; --speculative-algorithm; --speculative-draft-model-path; --speculative-num-steps; --speculative-eagle-topk; --speculative-num-draft-tokens; --speculative-attention-mode
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH
        cls.accuracy = 0.81
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)

        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "modelslim",
            "--disable-radix-cache",
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            QWEN3_32B_EAGLE3_WEIGHTS_PATH,
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "5",
            "--speculative-attention-mode",
            "decode",
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
        ]

        cls.extra_envs = {
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
        os.environ.update(cls.extra_envs)

    def test_gsm8k(self):
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=1500,
            other_args=[
                *self.common_args,
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=1319,
                max_new_tokens=512,
                parallel=128,
                host=f"http://{self.url.hostname}",
                port=int(self.url.port),
            )

            metrics = run_eval_few_shot_gsm8k(args)
            self.assertGreaterEqual(
                metrics["accuracy"],
                self.accuracy,
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
