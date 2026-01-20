import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestQwen3Next80B(GSM8KAscendMixin, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Next-80B-A3B-Instruct"
    accuracy = 0.92
    other_args = [
        "--tp-size",
        "4",
        "--disable-cuda-graph",
        "--attention-backend",
        "ascend",
        "--base-gpu-id",
        10,
        "--mem-fraction-static",
        0.8,
        "--disable-radix-cache",
    ]


if __name__ == "__main__":
    unittest.main()
