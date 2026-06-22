"""Llama-3.3-70B-Instruct GSM8K accuracy on Intel XPU (TP=8)."""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.gsm8k_xpu_mixin import GSM8KXPUMixin

register_xpu_ci(est_time=2700, suite="nightly-xpu-8-gpu", nightly=True)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestLlama3370BInstructXPU(GSM8KXPUMixin, CustomTestCase):
    model = "meta-llama/Llama-3.3-70B-Instruct"
    tp_size = 8
    accuracy = 0.85

    other_args = GSM8KXPUMixin.other_args + [
        "--max-total-tokens",
        "63356",
    ]


if __name__ == "__main__":
    unittest.main()
