import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import OLMOE_1B_7B_0924_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestOlMoe(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the allenai/OLMoE-1B-7B-0924 model on the GSM8K dataset is no less than 0.12.

    [Test Category] Model
    [Test Target] allenai/OLMoE-1B-7B-0924
    """

    model = OLMOE_1B_7B_0924_WEIGHTS_PATH
    accuracy = 0.12


if __name__ == "__main__":
    unittest.main()
