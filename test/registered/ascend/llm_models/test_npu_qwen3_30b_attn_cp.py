import os
import unittest
from types import SimpleNamespace

from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_npu_ci(est_time=500, suite="nightly-4-npu-a3", nightly=True)

QWEN3_30B_MODEL = QWEN3_30B_A3B_WEIGHTS_PATH
GSM8K_MIN_ACCURACY = 0.92
GSM8K_NUM_QUESTIONS = 100

_NPU_ENV_VARS = {
    "ASCEND_USE_FIA": "1",
}


class TestQwen330BAttnCP(CustomTestCase):
    """GSM8K accuracy test for Qwen3-30B-A3B mixed deployment on 4 NPUs.

    The test uses:
    - TP = 4
    - MOE_DP = 2
    - ATTN_CP = 2
    - prefill context parallel enabled

    This is the mixed/co-located deployment variant and reuses the Ascend
    environment variables from the PD GSM8K test.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.npu_env = {**os.environ, **_NPU_ENV_VARS}
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.7",
                "--max-running-requests",
                "32",
                "--attention-backend",
                "ascend",
                "--tp-size",
                "4",
                "--moe-dp-size",
                "2",
                "--attn-cp-size",
                "2",
                "--cuda-graph-max-bs",
                "32",
                "--enable-prefill-context-parallel",
            ],
            env=cls.npu_env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=GSM8K_NUM_QUESTIONS,
            max_new_tokens=512,
            parallel=32,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(
            "GSM8K accuracy "
            f"(mixed TP=4 MOE_DP=2 ATTN_CP=2, {GSM8K_NUM_QUESTIONS} samples): "
            f"{metrics['accuracy']:.3f}"
        )
        self.assertGreaterEqual(metrics["accuracy"], GSM8K_MIN_ACCURACY)


if __name__ == "__main__":
    unittest.main()
