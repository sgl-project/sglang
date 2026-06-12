import os
import torch
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ascend.test_mmlu import TestMMLU
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=200,
    suite="nightly-8-npu-a3",
    nightly=True,
)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestQwen3Next(GSM8KAscendMixin, TestMMLU, CustomTestCase):
    """
    Testcase:Test the Qwen3-Next-80B-A3B-Instruct-W8A8 model with DeepEP's low_latency mode enabled, and verify that
    there is no drop in accuracy compared to when DeepEP is not enabled.

    [Test Category] Parameter
    [Test Target] --moe-a2a-backend deepep, --deepep-mode low_latency
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--tp-size",
        8,
        "--mem-fraction-static",
        0.8,
        "--max-running-requests",
        80,
        "--watchdog-timeout",
        9000,
        "--disable-radix-cache",
        "--cuda-graph-bs",
        2,
        4,
        6,
        8,
        "--chunked-prefill-size",
        1024,
        "--max-prefill-tokens",
        28672,
        "--max-total-tokens",
        450560,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
    ]
    env = {
        # The product of the following two environment variables must be greater than --max-prefill-tokens
        # divide by dp size
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3000",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
        # In NPU scenarios, operators only support BF16 precision.
        # This environment variable needs to be set for quantizing weights.
        "SGLANG_DEEPEP_BF16_DISPATCH": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_ALGO": "level0:NA;level1:ring",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "160",
        "HCCL_BUFFSIZE": "2048",
        "GDN_ATTN_BACKEND_TRITON": "1",
        **os.environ,
    }

    # MMLU Configs
    mmlu_num_examples = 8
    accuracy_mmlu_threshold = 0.56  # MMLU accuracy ≥0.56

    # GSM8K Configs
    accuracy = 0.9  # GSM8K accuracy ≥0.9
    num_questions = 200
    gsm8k_num_shots = 5


if __name__ == "__main__":
    unittest.main()
