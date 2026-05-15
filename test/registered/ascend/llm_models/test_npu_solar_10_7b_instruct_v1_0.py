import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import SOLAR_10_7B_INSTRUCT_V1_0_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNpuSolar10_7bInstructV1_0(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the upstage/SOLAR-10.7B-Instruct-v1.0 model on the GSM8K dataset is no less than 0.7.

    [Test Category] Model
    [Test Target] upstage/SOLAR-10.7B-Instruct-v1.0
    """

    model = SOLAR_10_7B_INSTRUCT_V1_0_WEIGHTS_PATH
    accuracy = 0.7


if __name__ == "__main__":
    unittest.main()
