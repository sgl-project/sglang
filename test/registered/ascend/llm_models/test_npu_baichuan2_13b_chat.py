import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import BAICHUAN2_13B_CHAT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestBaichuan(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the baichuan-inc/Baichuan2-13B-Chat model on the GSM8K dataset is no less than 0.48.

    [Test Category] Model
    [Test Target] baichuan-inc/Baichuan2-13B-Chat
    """

    model = BAICHUAN2_13B_CHAT_WEIGHTS_PATH
    accuracy = 0.48
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--max-running-requests",
        "128",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
    ]
    gsm8k_num_shots = 1


if __name__ == "__main__":
    unittest.main()
