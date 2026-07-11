"""
Usage:
python3 -m unittest test_ascend_w4a4_quantization.TestAscendW4A4.test_gsm8k
"""

import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    ECO_TECH_QWEN3_32B_W4A4_LAOS_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="stage-b-test-4-npu-a3", nightly=False)
register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestAscendW4A4(GSM8KAscendMixin, CustomTestCase):

    model = ECO_TECH_QWEN3_32B_W4A4_LAOS_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "4",
        "--mem-fraction-static",
        "0.8",
        "--cuda-graph-bs",
        "64",
        "--disable-radix-cache",
    ]

    env = {
        **os.environ,
    }

    # GSM8K Configs
    accuracy = 0.80  # GSM8K accuracy ≥0.80
    num_questions = 1319
    gsm8k_num_shots = 5
    gsm8k_parallel = 64


if __name__ == "__main__":
    unittest.main()
