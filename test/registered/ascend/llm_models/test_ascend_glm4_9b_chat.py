import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import GLM_4_9B_CHAT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="run failed",
)


class TestGLM49BChat(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the ZhipuAI/glm-4-9b-chat model on the GSM8K dataset is no less than 0.79.

    [Test Category] Model
    [Test Target] ZhipuAI/glm-4-9b-chat
    """

    model = GLM_4_9B_CHAT_WEIGHTS_PATH
    accuracy = 0.79


if __name__ == "__main__":
    unittest.main()
