import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=300,
    suite="nightly-4-npu-a3",
    nightly=True,
)


class TestNPUEnableFp32LmHead(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy on the GSM8K dataset does not decrease after --enable-fp32-lm-head is set.

    [Test Category] Parameter
    [Test Target] --enable-fp32-lm-head
    """

    model = QWEN3_32B_WEIGHTS_PATH
    accuracy = 0.86
    other_args = [
        "--enable-fp32-lm-head",
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "4",
    ]


if __name__ == "__main__":
    unittest.main()
