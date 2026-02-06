import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import MIMO_7B_RL_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMiMo7BRL(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the XiaomiMiMo/MiMo-7B-RL model on the GSM8K dataset is no less than 0.75.

    [Test Category] Model
    [Test Target] XiaomiMiMo/MiMo-7B-RL
    """

    model = MIMO_7B_RL_WEIGHTS_PATH
    accuracy = 0.75


if __name__ == "__main__":
    unittest.main()
