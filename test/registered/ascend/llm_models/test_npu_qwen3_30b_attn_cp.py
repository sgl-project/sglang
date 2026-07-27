import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=500, suite="nightly-4-npu-a3", nightly=True)


class TestQwen330BAttnCP(GSM8KAscendMixin, CustomTestCase):
    """GSM8K accuracy test for Qwen3-30B-A3B mixed deployment on 4 NPUs.

    The test uses:
    - TP = 4
    - MOE_DP = 2
    - ATTN_CP = 2
    - prefill context parallel enabled

    This is the mixed/co-located deployment variant and reuses the Ascend
    environment variables from the PD GSM8K test.
    """

    model = QWEN3_30B_A3B_WEIGHTS_PATH
    other_args = [
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
        "--cuda-graph-max-bs-decode",
        "32",
        "--enable-prefill-context-parallel",
    ]

    env = {**os.environ, "ASCEND_USE_FIA": "1"}

    # GSM8K Configs
    accuracy = 0.92  # GSM8K accuracy ≥0.92
    gsm8k_parallel = 32
    num_questions = 100
    gsm8k_num_shots = 5


if __name__ == "__main__":
    unittest.main()
