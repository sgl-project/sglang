import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNpuEagle3(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify GSM8K inference accuracy ≥0.81 for model with specified EAGLE3 speculative inference parameters.

    [Test Category] Speculative Decoding
    [Test Target] --speculative-draft-model-quantization; --speculative-algorithm; --speculative-draft-model-path; --speculative-num-steps; --speculative-eagle-topk; --speculative-num-draft-tokens; --speculative-attention-mode
    """

    model = QWEN3_8B_WEIGHTS_PATH
    timeout_for_server_launch = 1500
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-radix-cache",
        "--speculative-draft-model-quantization",
        "unquant",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_8B_EAGLE3_WEIGHTS_PATH,
        "--speculative-num-steps",
        "4",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "5",
        "--speculative-attention-mode",
        "decode",
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.7",
        "--disable-cuda-graph",
        "--dtype",
        "bfloat16",
    ]

    env = {
        **os.environ,
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    }

    accuracy = 0.81
    num_questions = 1319


if __name__ == "__main__":
    unittest.main()
