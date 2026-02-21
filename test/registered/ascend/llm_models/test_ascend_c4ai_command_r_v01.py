import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    C4AI_COMMAND_R_V01_CHAT_TEMPLATE_PATH,
    C4AI_COMMAND_R_V01_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestC4AI(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the CohereForAI/c4ai-command-r-v01 model on the GSM8K dataset is no less than 0.55.

    [Test Category] Model
    [Test Target] CohereForAI/c4ai-command-r-v01
    """

    model = C4AI_COMMAND_R_V01_WEIGHTS_PATH
    accuracy = 0.55
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--chat-template",
        C4AI_COMMAND_R_V01_CHAT_TEMPLATE_PATH,
        "--tp-size",
        "2",
        "--dtype",
        "bfloat16",
    ]


if __name__ == "__main__":
    unittest.main()
