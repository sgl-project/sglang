import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import LLAMA_2_7B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestLlama(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the LLM-Research/Llama-2-7B model on the GSM8K dataset is no less than 0.18.

    [Test Category] Model
    [Test Target] LLM-Research/Llama-2-7B
    """

    model = LLAMA_2_7B_WEIGHTS_PATH
    accuracy = 0.18


if __name__ == "__main__":
    unittest.main()
