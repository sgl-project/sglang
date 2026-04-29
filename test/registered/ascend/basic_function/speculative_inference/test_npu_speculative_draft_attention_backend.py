import logging
import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)

MODEL_PATH = DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH


class TestAscendSpeculativeDraftAttentionAndMoeRunner(CustomTestCase):
    """Testcase: Test configuration '--speculative-draft-attention-backend' and '--speculative-moe-runner-backend' on the GSM8K dataset is no less than 0.9.

    [Test Category] Parameter
    [Test Target] --speculative-draft-attention-backend; --speculative-moe-runner-backend
    """

    @classmethod
    def setUpClass(cls):
        os.environ["HCCL_BUFFSIZE"] = "2048"
        os.environ["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"
        os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
        cls.env = os.environ.copy()
        cls.models = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)

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
            MODEL_PATH,
            cls.base_url,
            timeout=1500,
            other_args=cls.common_args,
            env=cls.env,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            eval_name="gsm8k",
            api="completion",
            num_examples=1319,
            num_threads=128,
            max_new_tokens=512,
            num_shots=5,
        )

        metrics = run_eval(args)
        score = metrics["score"]
        logger.info(f"GSM8K score for {MODEL_PATH}: {score:.4f}")
        self.assertIsNotNone(score, "GSM8K evaluation returned no score")
        self.assertIsInstance(score, float, "Score should be a float")
        self.assertGreaterEqual(score, 0.9, f"GSM8K score {score} below threshold 0.9")


if __name__ == "__main__":
    unittest.main()
