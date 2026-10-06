"""Qwen3-30B-A3B GSM8K accuracy on Intel XPU with DeepEP MoE all-to-all (TP=EP=4).

Uses deep_ep_xpu intranode normal dispatch/combine over PCIe P2P.
Scored by ``simple_eval_gsm8k.GSM8KEval``.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

# deep_ep_xpu is not yet in the public XPU CI image; enable once it is installed.
register_xpu_ci(
    est_time=2400,
    suite="nightly-xpu-4-gpu",
    nightly=True,
    disabled="deep_ep_xpu not installed in XPU CI image",
)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestQwen3_30BA3BDeepEPXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "Qwen/Qwen3-30B-A3B"
    tp_size = 4
    accuracy = 0.90
    timeout_for_server_launch = 3600
    num_examples = 50
    num_threads = 4
    max_tokens = 8192

    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--ep-size",
        "4",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--max-total-tokens",
        "65536",
        "--mem-fraction-static",
        "0.8",
    ]


if __name__ == "__main__":
    unittest.main()
