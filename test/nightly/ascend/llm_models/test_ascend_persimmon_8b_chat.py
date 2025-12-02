import unittest

from gsm8k_ascend_mixin import GSM8KAscendMixin

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMistral7B(GSM8KAscendMixin, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/Howeee/persimmon-8b-chat"
    accuracy = 0.17


if __name__ == "__main__":
    unittest.main()
