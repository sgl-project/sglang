import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestGrok2(GSM8KAscendMixin, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/huihui-ai/grok-2"
    accuracy = 0.91
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--tokenizer-path",
        "/root/.cache/modelscope/hub/models/huihui-ai/grok-2/tokenizer.tok.json",
        "--tp-size",
        "16",
    ]


if __name__ == "__main__":
    unittest.main()
