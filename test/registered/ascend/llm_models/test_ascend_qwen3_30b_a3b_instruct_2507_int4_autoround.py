import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_INT4_AUTOROUND_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="per-commit-1-npu-a2")


class TestQwen330BA3BInstruct2507Int4AutoRound(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Intel/Qwen3-30B-A3B-Instruct-2507-int4-AutoRound model on the GSM8K dataset is no less than 0.85.

    [Test Category] Model
    [Test Target] Intel/Qwen3-30B-A3B-Instruct-2507-int4-AutoRound (MOE)
    """

    model = QWEN3_30B_A3B_INSTRUCT_2507_INT4_AUTOROUND_WEIGHTS_PATH
    accuracy = 0.85
    gsm8k_num_questions = 1319
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--quantization",
        "auto-round",
    ]


if __name__ == "__main__":
    unittest.main()
