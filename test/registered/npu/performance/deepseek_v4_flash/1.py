import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="base-b-test-8-npu-a3", nightly=False)
register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestQwen330Bw8a8FuseEP(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen3-30B-A3B-w8a8 model on the GSM8K dataset is no less than 0.90.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-30B-A3B-w8a8
    """

    model = QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
    accuracy = 0.90
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.8,
        "--attention-backend",
        "ascend",
        "--cuda-graph-max-bs-decode",
        128,
        "--tp-size",
        4,
        "--moe-a2a-backend",
        "ascend_fuseep",
        "--fuseep-mode",
        2,
    ]

    env = {
        **os.environ,
        "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "188416",
    }


class TestQwen330Bw8a8FuseEPTestQwen330Bw8a8FuseModeWithOne(
    GSM8KAscendMixin, CustomTestCase
):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen3-30B-A3B-w8a8 model on the GSM8K dataset is
    no less than 0.90 with --fuseep-mode = 1.

    [Test Category] Parameter
    [Test Target] ---fuseep-mode
    """

    model = QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
    accuracy = 0.90
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.8,
        "--attention-backend",
        "ascend",
        "--cuda-graph-max-bs-decode",
        128,
        "--tp-size",
        8,
        "--ep-size",
        8,
        "--moe-a2a-backend",
        "ascend_fuseep",
        "--fuseep-mode",
        1,
        "--max-prefill-tokens",
        256,
        "--chunked-prefill-size",
        256,
    ]

    env = {
        **os.environ,
        "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
        "HCCL_BUFFSIZE": "1024",
    }


if __name__ == "__main__":
    unittest.main()
