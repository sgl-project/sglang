import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import ORION_14B_BASE_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestAFM(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the OrionStarAI/Orion-14B-Base model on the GSM8K dataset is no less than 0.38.

    [Test Category] Model
    [Test Target] OrionStarAI/Orion-14B-Base
    """

    model = ORION_14B_BASE_WEIGHTS_PATH
    accuracy = 0.38


if __name__ == "__main__":
    unittest.main()
