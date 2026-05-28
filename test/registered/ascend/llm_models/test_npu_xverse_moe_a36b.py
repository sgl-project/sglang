import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import XVERSE_MOE_A36B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class TestXverse(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the xverse/XVERSE-MoE-A36B model on the GSM8K dataset is no less than 0.24.

    [Test Category] Model
    [Test Target] xverse/XVERSE-MoE-A36B
    """

    model = XVERSE_MOE_A36B_WEIGHTS_PATH
    accuracy = 0.24
    timeout_for_server_launch = 3000
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        16,
        "--context-length",
        2048,
    ]


if __name__ == "__main__":
    unittest.main()
