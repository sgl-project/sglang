import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import KIMI_LINEAR_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="run failed",
)



class TestKimiLinear(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the moonshotai/Kimi-Linear-48B-A3B-Instruct model on the GSM8K dataset is no less than 0.88.

    [Test Category] Model
    [Test Target] moonshotai/Kimi-Linear-48B-A3B-Instruct
    """

    model = KIMI_LINEAR_WEIGHTS_PATH
    accuracy = 0.88
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "2",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--max-running-requests",
        "16",
    ]


if __name__ == "__main__":
    unittest.main()
