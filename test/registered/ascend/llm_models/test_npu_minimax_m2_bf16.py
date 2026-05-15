import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import MINIMAX_M2_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


class TestMiniMaxM2(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the cyankiwi/MiniMax-M2-BF16 model on the GSM8K dataset is no less than 0.9.

    [Test Category] Model
    [Test Target] cyankiwi/MiniMax-M2-BF16
    """

    model = MINIMAX_M2_WEIGHTS_PATH
    accuracy = 0.9
    timeout_for_server_launch = 3000
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "8",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--max-running-requests",
        "64",
        "--chunked-prefill-size",
        "-1",
    ]


if __name__ == "__main__":
    unittest.main()
