import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    Qwen3_30B_A3B_Instruct_2507_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestQwen330B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen3-30B-A3B-Instruct-2507 model on the GSM8K dataset is no less than 0.90.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-30B-A3B-Instruct-2507
    """

    model = Qwen3_30B_A3B_Instruct_2507_WEIGHTS_PATH
    accuracy = 0.90
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.7,
        "--max-running-requests",
        32,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--cuda-graph-max-bs",
        32,
        "--tp-size",
        2,
    ]


if __name__ == "__main__":
    unittest.main()
