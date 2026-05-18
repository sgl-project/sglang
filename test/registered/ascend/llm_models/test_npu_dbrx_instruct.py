import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import DBRX_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


class TestDbrx(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the AI-ModelScope/dbrx-instruct model on the GSM8K dataset is no less than 0.735.

    [Test Category] Model
    [Test Target] AI-ModelScope/dbrx-instruct
    """

    model = DBRX_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.735
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "8",
    ]


if __name__ == "__main__":
    unittest.main()
