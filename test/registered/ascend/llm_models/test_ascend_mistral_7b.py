import unittest

from gsm8k_ascend_mixin import GSM8KAscendMixin

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMistral7B(GSM8KAscendMixin, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/mistralai/Mistral-7B-Instruct-v0.2"
    accuracy = 0.375


if __name__ == "__main__":
    unittest.main()
