import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="run failed",
)


class TestQwen3Coder480BA35B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot model on the GSM8K dataset is no less than 0.94.

    [Test Category] Model
    [Test Target] Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot
    """

    model = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_WEIGHTS_PATH
    accuracy = 0.94
    timeout_for_server_launch = 3000
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "16",
        "--quantization",
        "modelslim",
    ]


if __name__ == "__main__":
    unittest.main()
