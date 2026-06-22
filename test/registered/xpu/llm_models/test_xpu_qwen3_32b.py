"""Qwen3-32B GSM8K accuracy on Intel XPU (TP=4)."""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.gsm8k_xpu_mixin import GSM8KXPUMixin

register_xpu_ci(est_time=1800, suite="nightly-xpu-4-gpu", nightly=True)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestQwen3_32BXPU(GSM8KXPUMixin, CustomTestCase):
    model = "Qwen/Qwen3-32B"
    tp_size = 4
    accuracy = 0.85

    other_args = GSM8KXPUMixin.other_args + [
        "--max-total-tokens",
        "65536",
        "--mem-fraction-static",
        "0.8",
    ]


if __name__ == "__main__":
    unittest.main()
