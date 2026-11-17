"""Qwen3.5-35B-A3B GSM8K accuracy on Intel XPU (TP=4).

Scored by sgl-eval ``gsm8k`` with sampled thinking.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=3300, suite="nightly-xpu-kernel-main-4-gpu", nightly=True)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestQwen3_5_35BA3BXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "Qwen/Qwen3.5-35B-A3B"
    tp_size = 4
    # Lowest score over 2 runs (0.85, 0.90) on B70; raise only on new data.
    accuracy = 0.85
    timeout_for_server_launch = 3600
    num_examples = 20
    num_threads = 4
    # Greedy thinking loops re-checking sgl-eval's boxed answer, and
    # non-thinking skips the CoT (0.42); sampling ends most loops.
    temperature = 1.0
    top_p = 0.95
    max_tokens = 16384

    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--page-size",
        "128",
        "--max-total-tokens",
        "131072",
        "--mem-fraction-static",
        "0.85",
    ]


if __name__ == "__main__":
    unittest.main()
