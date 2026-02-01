import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="run failed",
)


class TestQwen332B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen3-32B model on the GSM8K dataset is no less than 0.88.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-32B
    """

    model = QWEN3_32B_WEIGHTS_PATH
    accuracy = 0.88
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "4",
    ]


if __name__ == "__main__":
    unittest.main()
