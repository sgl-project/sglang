import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)

TEST_MODEL_MATRIX = {
    DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH: {
        "accuracy": 0.90,
    },
}


class TestAscendSpeculativeAcceptThreshold(CustomTestCase):
    """Testcase: Test configuration '--speculative-draft-attention-backend' and '--speculative-moe-runner-backend' on the GSM8K dataset is no less than 0.9.

    [Test Category] Parameter
    [Test Target] --speculative-draft-attention-backend; --speculative-moe-runner-backend
    """

    os.environ["HCCL_BUFFSIZE"] = "2048"
    os.environ["SGLANG_ENABLE_OVERLAP_PLAN_SITEAM"] = "1"
    os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
    env = os.environ.copy()

    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "modelslim",
            "--mem-fraction-static",
            0.7,
            "--disable-radix-cache",
            "--chunked-prefill-size",
            32768,
            "--tp-size",
            16,
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            1,
            "--speculative-eagle-topk",
            1,
            "--speculative-num-draft-tokens",
            2,
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "auto",
            "--max-running-requests",
            64,
            "--speculative-draft-attention-backend",
            "ascend",
            "--speculative-moe-runner-backend",
            "auto",
        ]

        cls.process = popen_launch_server(
            DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH,
            cls.base_url,
            timeout=1500,
            other_args=cls.common_args,
            env=cls.env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
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
            TEST_MODEL_MATRIX[DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH][
                "accuracy"
            ],
        )


if __name__ == "__main__":
    unittest.main()
