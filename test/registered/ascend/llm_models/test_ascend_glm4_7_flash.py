import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import GLM_4_7_FLASH_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="run failed",
)


class TestGLM47Flash(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the ZhipuAI/CLM-4.7-Flash model on the GSM8K dataset is no less than 0.78.

    [Test Category] Model
    [Test Target] ZhipuAI/GLM-4.7-Flash
    """

    model = GLM_4_7_FLASH_WEIGHTS_PATH
    accuracy = 0.78
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "4",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--max-running-requests",
        "64",
        "--chunked-prefill-size",
        "-1",
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm45",
        "--chat-template",
        f"{model}/chat_template.jinja",
        "--served-model-name",
        "glm47Flash",
    ]


if __name__ == "__main__":
    os.environ["SGLANG_NPU_FORWARD_NATIVE_TOPK"] = "1"
    unittest.main()
