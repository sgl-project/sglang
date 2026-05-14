import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestDbrx(GSM8KAscendMixin, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/AI-ModelScope/dbrx-instruct"
    accuracy = 0.735
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "8",
    ]


if __name__ == "__main__":
    unittest.main()
