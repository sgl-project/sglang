import unittest

from gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.test_utils import CustomTestCase


class TestGLM49BChat(GSM8KAscendMixin, CustomTestCase):
  model = "/root/.cache/modelscope/hub/models/ZhipuAI/glm-4-9b-chat"
  accuracy = 0.00

if __name__ == "__main__":
    unittest.main()
