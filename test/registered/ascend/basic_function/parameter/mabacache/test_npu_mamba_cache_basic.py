import unittest
from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestMambaCache(GSM8KAscendMixin, CustomTestCase):
    """Testcase：Verify that the accuracy use mamba cache on the gsm8k dataset is no less than 0.92.

    [Test Category] Parameter
    [Test Target] --mem-fraction-static, --mamba-ssm-dtype, --mamba-full-memory-ratio
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST
    accuracy = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.gsm8k_accuracy
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.5",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--max-mamba-cache-size",
        "None",
        "--mamba-ssm-dtype",
        "float32",
        "--mamba-full-memory-ratio",
        "0.9",
        "--mamba-scheduler-strategy",
        "auto",
        "--mamba-track-interval",
        "256",
        "--tp-size",
        8,
        "--disable-radix-cache"
    ]


if __name__ == "__main__":
    unittest.main()
