import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="https://github.com/Ascend/sglang/issues/25",
)


class TestLlama4(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the meta-llama/Llama-4-Scout-17B-16E-Instruct model on the GSM8K dataset is no less than 0.9.

    [Test Category] Model
    [Test Target] meta-llama/Llama-4-Scout-17B-16E-Instruct
    """

    model = LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.9
    other_args = [
        "--chat-template",
        "llama-4",
        "--tp-size",
        4,
        "--mem-fraction-static",
        "0.9",
        "--context-length",
        "8192",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]


if __name__ == "__main__":
    unittest.main()
