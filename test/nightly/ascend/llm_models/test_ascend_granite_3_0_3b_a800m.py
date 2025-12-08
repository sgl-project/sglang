import unittest

from gsm8k_ascend_mixin import GSM8KAscendMixin

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMistral7B(GSM8KAscendMixin, CustomTestCase):
    model = (
        "/root/.cache/modelscope/hub/models/ibm-granite/granite-3.0-3b-a800m-instruct"
    )
    accuracy = 0.00


if __name__ == "__main__":
    unittest.main()
