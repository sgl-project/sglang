import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    GROK_2_WEIGHTS_PATH,
    GROK_2_WEIGHTS_TOKENIZER_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="full-16-npu-a3",
    nightly=False,
)


class TestGrok2(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the huihui-ai/grok-2 model on the GSM8K dataset is no less than 0.91.

    [Test Category] Model
    [Test Target] huihui-ai/grok-2
    """

    model = GROK_2_WEIGHTS_PATH
    accuracy = 0.91
    timeout_for_server_launch = 3000
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--tokenizer-path",
        GROK_2_WEIGHTS_TOKENIZER_PATH,
        "--tp-size",
        "16",
    ]


if __name__ == "__main__":
    unittest.main()
