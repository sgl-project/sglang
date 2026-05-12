import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ascend.test_mmlu import TestMMLU
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestDeepEpDeepseekV32(GSM8KAscendMixin, TestMMLU, CustomTestCase):
    """Testcase: Verify that for the DeepSeek V3.2 model in the single-machine colocation scenario,
    its inference accuracy on the MMLU and GSM8K dataset meets the preset standard when the parameter --deepep-mode auto is configured.

    [Test Category] Expert Parallelism
    [Test Target] --moe-a2a-backend deepep;--deepep-mode
    """

    model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH

    timeout_for_server_launch = 60000
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "16",
        "--quantization",
        "modelslim",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--mem-fraction-static",
        0.82,
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--context-length",
        40960,
        "--max-prefill-tokens",
        40960,
        "--max-total-tokens",
        40960,
    ]

    env = {
        **os.environ,
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "HCCL_BUFFSIZE": "1600",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "SGLANG_NPU_USE_MLAPO": "0",
        "SGLANG_NPU_USE_MULTI_STREAM": "1",
        "TASK_QUEUE_ENABLE": "0",
    }

    accuracy = 0.95  # Test GSM8K accuracy ≥0.95
    accuracy_mmlu = 0.85  # Test MMLU accuracy ≥0.85


if __name__ == "__main__":
    unittest.main()
