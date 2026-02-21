import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestDeepseekR1Nvfp4CuteDSLDeepEP(CustomTestCase):
    """Testcase: Test configure `--enable-single-batch-overlap`, use the GSM8K dataset, and ensure an inference accuracy of at least 0.86.

    [Test Category] Parameter
    [Test Target] --enable-single-batch-overlap
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_32B_WEIGHTS_PATH

        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.7",
            "--max-running-requests",
            "16",
            "--chunked-prefill-size",
            "512",
            "--tp",
            "16",
            "--attention-backend",
            "ascend",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "auto",
            "--disable-cuda-graph",
            "--enable-single-batch-overlap",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3000,
            other_args=other_args,
            env={
                **os.environ,
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "20",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            parallel=512,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(metrics["accuracy"], 0.86)


if __name__ == "__main__":
    unittest.main()
