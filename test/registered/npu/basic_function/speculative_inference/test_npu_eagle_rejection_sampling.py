import os
import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import QWEN3_5_9B_MODEL_PATH
from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNpuNextnRejectionSampling(GSM8KAscendMixin, CustomTestCase):
    """Verify GSM8K inference accuracy and speculative acceptance length for NEXTN
    with --speculative-use-rejection-sampling on NPU.

    NEXTN shares the target's full vocab (unlike Qwen EAGLE3's reduced hot vocab),
    so it is the compatible path for classic rejection sampling.

    [Test Category] Speculative Decoding
    [Test Target] --speculative-use-rejection-sampling; --speculative-algorithm; --speculative-num-steps; --speculative-eagle-topk; --speculative-num-draft-tokens
    """

    model = QWEN3_5_9B_MODEL_PATH
    timeout_for_server_launch = 1500
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-radix-cache",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-use-rejection-sampling",
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.7",
        "--disable-cuda-graph",
        "--dtype",
        "bfloat16",
        "--mamba-ssm-dtype",
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