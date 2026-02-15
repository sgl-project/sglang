import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import TRINITY_MINI_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="run failed",
)


class TestTrinityMini(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the arcee-ai/Trinity-Mini model on the GSM8K dataset is no less than 0.85.

    [Test Category] Model
    [Test Target] arcee-ai/Trinity-Mini
    """

    model = TRINITY_MINI_WEIGHTS_PATH
    accuracy = 0.85
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "2",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--context-length",
        "4096",
        "--max-running-requests",
        "64",
        "--chunked-prefill-size",
        "-1",
        "--chat-template",
        f"{TRINITY_MINI_WEIGHTS_PATH}/chat_template.jinja",
    ]


if __name__ == "__main__":
    os.environ["SGLANG_NPU_FORWARD_NATIVE_TOPK"]="1"
    unittest.main()
