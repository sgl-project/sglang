"""Gemma-4-E4B GSM8K accuracy on Intel XPU (TP=2).

Scored by ``simple_eval_gsm8k.GSM8KEval`` (the same evaluator AMD and
NVIDIA nightlies use); threshold mirrors theirs.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=900, suite="nightly-xpu-2-gpu", nightly=True)


@unittest.skip("Not yet validated end-to-end on intel-bmg-nightly; re-enable once a passing run is recorded.")
@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestGemma4E4BXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "google/gemma-4-E4B"
    tp_size = 2
    accuracy = 0.70

    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--max-total-tokens",
        "65536",
        "--mem-fraction-static",
        "0.9",
        "--page-size",
        "64",
    ]


if __name__ == "__main__":
    unittest.main()
