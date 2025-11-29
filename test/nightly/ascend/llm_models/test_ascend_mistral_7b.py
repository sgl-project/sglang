import unittest

from gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.test_utils import CustomTestCase


class TestMistral7B(GSM8KAscendMixin, CustomTestCase):
  model = "/root/.cache/modelscope/hub/models/mistralai/Mistral-7B-Instruct-v0.2"
  accuracy = 0.00

if __name__ == "__main__":
    unittest.main()
