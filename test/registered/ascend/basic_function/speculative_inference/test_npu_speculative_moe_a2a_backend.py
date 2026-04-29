import logging
import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH,
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


class TestAscendSpeculativeMoeA2ABackend(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.accuracy = 0.95
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.env = os.environ.copy()
        cls.env.update({
            "HCCL_BUFFSIZE": "2048",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_NPU_FUSED_MOE_MODE": "1",
        })
        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "modelslim",
            "--mem-fraction-static",
            0.8,
            "--disable-radix-cache",
            "--chunked-prefill-size",
            2048,
            "--tp-size",
            16,
            "--disable-cuda-graph",
            "--speculative-moe-a2a-backend",
            "ascend_fuseep",
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            1,
            "--speculative-eagle-topk",
            1,
            "--speculative-num-draft-tokens",
            2,
            "--moe-a2a-backend",
            "ascend_fuseep",
            "--deepep-mode",
            "auto",
        ]

    def test_a_gsm8k(self):
        logger.info(f"##=== Testing accuracy: {self.model} ===##")
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=1500,
            other_args=self.common_args,
            env=self.env,
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                eval_name="gsm8k",
                api="completion",
                num_examples=1319,
                num_threads=128,
                max_new_tokens=512,
                num_shots=5,
                temperature=0.0,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(
                metrics["score"],
                self.accuracy,
                f"GSM8K score {metrics['score']} below threshold {self.accuracy}",
            )
        finally:
            if process is not None:
                kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
