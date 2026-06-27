"""Qwen3-30B-A3B GSM8K accuracy on Intel XPU (TP=4).

Requires ``SGLANG_USE_SGL_XPU=1`` so the MoE path uses the Intel-XPU-aware
implementation; passed via the mixin's ``env`` rather than the shell so
the test is reproducible from a clean environment.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.gsm8k_xpu_mixin import GSM8KXPUMixin

register_xpu_ci(est_time=1500, suite="nightly-xpu-4-gpu", nightly=True)


@unittest.skip("Not yet validated end-to-end on intel-bmg-nightly; re-enable once a passing run is recorded.")
@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestQwen3_30BA3BXPU(GSM8KXPUMixin, CustomTestCase):
    model = "Qwen/Qwen3-30B-A3B"
    tp_size = 4
    accuracy = 0.86

    env = {"SGLANG_USE_SGL_XPU": "1"}

    other_args = GSM8KXPUMixin.other_args + [
        "--max-total-tokens",
        "65536",
        "--mem-fraction-static",
        "0.8",
    ]


if __name__ == "__main__":
    unittest.main()
