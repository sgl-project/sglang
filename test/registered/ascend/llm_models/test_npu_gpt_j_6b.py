import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import GPT_J_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)


class TestAFM(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the EleutherAI/gpt-j-6b model on the GSM8K dataset is no less than 0.037.

    [Test Category] Model
    [Test Target] EleutherAI/gpt-j-6b
    """

    model = GPT_J_6B_WEIGHTS_PATH
    accuracy = 0.037
    other_args = [
        "--trust-remote-code",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--tp-size",
        2,
        "--mem-fraction-static",
        0.8,
        "--dtype",
        "bfloat16",
        "--enable-multimodal",
    ]


if __name__ == "__main__":
    unittest.main()
