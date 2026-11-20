"""Online FP8 (``--quantization fp8``) GSM8K accuracy on Intel XPU.

The bf16 checkpoint is quantized at load. Dense layers run weight-only FP8 (W8A16,
``sgl_kernel.fp8_linear``) where sgl-kernel-xpu supports it and the W8A8 path
elsewhere; the accuracy bar is the same for both.
Radix cache and overlap scheduling stay at their defaults.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=300, suite="stage-b-test-1-gpu-xpu")


@unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
class TestXPUOnlineFp8GSM8K(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
    # 50 questions on Arc Pro B65: fp8_linear 0.66, W8A8 path 0.62 (standard error
    # ~0.07).
    accuracy = 0.50
    num_examples = 50
    other_args = [
        "--device",
        "xpu",
        "--attention-backend",
        "intel_xpu",
        "--dtype",
        "bfloat16",
        "--quantization",
        "fp8",
        "--mem-fraction-static",
        "0.6",
    ]


if __name__ == "__main__":
    unittest.main()
