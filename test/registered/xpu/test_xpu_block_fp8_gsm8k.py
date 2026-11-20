"""Block-FP8 checkpoint (128x128 ``weight_block_size``) GSM8K accuracy on Intel XPU.

Loads a pre-quantized block-FP8 checkpoint, so the weight loader shards the block
scales. Dense layers run weight-only FP8 (W8A16, ``sgl_kernel.fp8_linear``) with the
stored block scales where sgl-kernel-xpu supports it, and the existing block-FP8 path
elsewhere; the accuracy bar is the same for both.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_QWEN_FP8,
    CustomTestCase,
)
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=300, suite="stage-b-test-1-gpu-xpu")


@unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
class TestXPUBlockFp8GSM8K(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = DEFAULT_MODEL_NAME_FOR_TEST_QWEN_FP8
    # 50 questions on Arc Pro B65: fp8_linear 0.72, existing block-FP8 path 0.74
    # (standard error ~0.065).
    accuracy = 0.60
    num_examples = 50
    # Qwen3 thinks past max_tokens on every question; answer directly.
    chat_template_kwargs = {"enable_thinking": False}
    other_args = [
        "--device",
        "xpu",
        "--attention-backend",
        "intel_xpu",
        "--mem-fraction-static",
        "0.6",
    ]


if __name__ == "__main__":
    unittest.main()
