import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestQwen306B(GSM8KAscendMixin, CustomTestCase):
    model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    accuracy = 0.38
    other_args = [
        "--chunked-prefill-size",
        256,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]


if __name__ == "__main__":
    unittest.main()
