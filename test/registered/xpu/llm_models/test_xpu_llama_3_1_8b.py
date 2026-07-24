"""Llama-3.1-8B-Instruct GSM8K accuracy on Intel XPU (TP=2).

TP=4 wedges the Level Zero driver during the first prefill batch on Arc/BMG;
TP=2 runs cleanly with the same model and serves at ~18 tok/s.

Scored by ``simple_eval_gsm8k.GSM8KEval``.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=1200, suite="nightly-xpu-2-gpu", nightly=True)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestLlama31_8BInstructXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "meta-llama/Llama-3.1-8B-Instruct"
    tp_size = 2
    accuracy = 0.80

    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--max-total-tokens",
        "65536",
        "--mem-fraction-static",
        "0.8",
    ]


if __name__ == "__main__":
    unittest.main()
