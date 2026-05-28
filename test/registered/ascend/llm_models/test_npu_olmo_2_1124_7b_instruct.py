import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import OLMO_2_1124_7B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNpuOlmo2_1124_7bInstruct(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the allenai/OLMo-2-1124-7B-Instruct model on the GSM8K dataset is no less than 0.74.

    [Test Category] Model
    [Test Target] allenai/OLMo-2-1124-7B-Instruct

    """

    model = OLMO_2_1124_7B_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.74


if __name__ == "__main__":
    unittest.main()
