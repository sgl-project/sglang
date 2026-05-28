import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestQwen3Next80B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen3-Next-80B-A3B-Instruct model on the GSM8K dataset is no less than 0.92.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-Next-80B-A3B-Instruct
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.92
    other_args = [
        "--tp-size",
        "4",
        "--disable-cuda-graph",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        0.8,
        "--disable-radix-cache",
    ]


if __name__ == "__main__":
    unittest.main()
