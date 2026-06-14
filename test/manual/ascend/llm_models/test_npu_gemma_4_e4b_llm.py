import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import GEMMA_4_E4B_WEIGHTS_PATH
from sglang.test.test_utils import CustomTestCase


class TestGemma4E4B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the google/gemma-4-E4B-it model on the GSM8K dataset is no less than 0.60.

    [Test Category] Model
    [Test Target] google/gemma-4-E4B-it
    """

    model = GEMMA_4_E4B_WEIGHTS_PATH
    accuracy = 0.60
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.7",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "1",
    ]


if __name__ == "__main__":
    unittest.main()
