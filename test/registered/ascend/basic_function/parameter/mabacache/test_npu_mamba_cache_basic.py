import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestMambaCache(GSM8KAscendMixin, CustomTestCase):
    """Testcase：Verify the MambaCache

    [Test Category] Parameter
    [Test Target] --lora-target-modules
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.model_path
    accuracy = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.gsm8k_accuracy
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.5",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mamba-ssm-dtype",
        "float32",
        "--mamba-full-memory-ratio",
        "0.9",
        "--mamba-scheduler-strategy",
        "auto",
        "--mamba-track-interval",
        "256",
        "--tp-size",
        "8",
        "--disable-radix-cache",
        "--base-gpu-id",
        "8",
    ]


if __name__ == "__main__":
    unittest.main()
