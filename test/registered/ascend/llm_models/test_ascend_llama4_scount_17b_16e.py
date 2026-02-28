import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestLlama4(GSM8KAscendMixin, CustomTestCase):
    model = (
        "/root/.cache/modelscope/hub/models/meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )
    accuracy = 0.9
    other_args = [
        "--chat-template",
        "llama-4",
        "--tp-size",
        4,
        "--mem-fraction-static",
        "0.9",
        "--context-length",
        "8192",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]


if __name__ == "__main__":
    unittest.main()
