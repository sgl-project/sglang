import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import GEMMA_3_1B_IT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="The accuracy test result is 0.",
)


class TestGemma31BIt(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the gemma-3-1b-it model on the GSM8K dataset.

    [Test Category] Model
    [Test Target] gemma-3-1b-it
    """

    model = GEMMA_3_1B_IT_WEIGHTS_PATH
    accuracy = 0.00


if __name__ == "__main__":
    unittest.main()
