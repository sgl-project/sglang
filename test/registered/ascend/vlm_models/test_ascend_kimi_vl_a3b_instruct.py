import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestKimiVLA3BInstruct(GSM8KAscendMixin, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/Kimi/Kimi-VL-A3B-Instruct"
    accuracy = 0.66
    other_args = [
        "--trust-remote-code",
        "--max-running-requests",
        2048,
        "--mem-fraction-static",
        0.7,
        "--attention-backend",
        "ascend",
        "--tp-size",
        "4",
        "--disable-cuda-graph",
    ]


if __name__ == "__main__":
    unittest.main()
