import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import LING_LITE_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestLingLite(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the inclusionAI/Ling-lite model on the GSM8K dataset is no less than 0.75.

    [Test Category] Model
    [Test Target] inclusionAI/Ling-lite
    """

    model = LING_LITE_WEIGHTS_PATH
    accuracy = 0.75
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "2",
    ]


if __name__ == "__main__":
    unittest.main()
