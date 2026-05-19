import os
import torch
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_WEIGHTS_PATH,
)
from sglang.test.ascend.test_mmlu import TestMMLU
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=200, suite="nightly-16-npu-a3", nightly=True)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestDeepEpQwen(GSM8KAscendMixin, TestMMLU, CustomTestCase):
    """
    Testcase:Test the Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot model with DeepEP's auto mode enabled,
    and verify that there is no drop in accuracy compared to when DeepEP is not enabled.

    [Test Category] Expert Parallelism
    [Test Target] --moe-a2a-backend, --deepep-mode
    """

    model = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--max-running-requests",
        96,
        "--context-length",
        8192,
        "--dtype",
        "bfloat16",
        "--chunked-prefill-size",
        28672,
        "--max-prefill-tokens",
        458880,
        "--disable-radix-cache",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--tp-size",
        16,
        "--dp-size",
        4,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mem-fraction-static",
        0.7,
        "--cuda-graph-bs",
        16,
        20,
        24,
    ]
    env = {
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "HCCL_BUFFSIZE": "2100",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "TRANSFORMERS_VERBOSITY": "error",
        **os.environ,
    }

    # MMLU Configs
    mmlu_num_examples = 8
    accuracy_mmlu_threshold = 0.61  # MMLU accuracy ≥0.61

    # GSM8K Configs
    accuracy = 0.91  # GSM8K accuracy ≥0.91
    num_questions = 200
    gsm8k_num_shots = 8


if __name__ == "__main__":
    unittest.main()
