import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_MODELSLIM_INT4_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="per-commit-2-npu-a2")


class TestQwen317BGPTQInt8(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Eco-Tech/Qwen3-30B-A3B-w4a4-LAOS model on the GSM8K dataset is no less than 0.85.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-1.7B-GPTQ-Int8
    """

    model = QWEN3_30B_MODELSLIM_INT4_WEIGHTS_PATH
    accuracy = 0.85
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.8,
        "--max-running-requests",
        32,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--cuda-graph-max-bs",
        32,
        "--tp-size",
        2,
    ]


if __name__ == "__main__":
    unittest.main()
