import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import STABLELM_2_1_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestStablelm(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the stabilityai/stablelm-2-1_6b model on the GSM8K dataset is no less than 0.195.

    [Test Category] Model
    [Test Target] stabilityai/stablelm-2-1_6b
    """

    model = STABLELM_2_1_6B_WEIGHTS_PATH
    accuracy = 0.195
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        1,
        "--enable-torch-compile",
    ]


if __name__ == "__main__":
    unittest.main()
