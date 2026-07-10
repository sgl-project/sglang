"""NVIDIA-Nemotron-3-Nano-30B-A3B GSM8K accuracy on Intel XPU (TP=4).

Scored by ``simple_eval_gsm8k.GSM8KEval`` (the same evaluator AMD and
NVIDIA nightlies use).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=2400, suite="nightly-xpu-4-gpu", nightly=True)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestNemotron3Nano30BA3BXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    tp_size = 4
    accuracy = 0.72
    timeout_for_server_launch = 3600
    # Generation cap for the GSM8K eval (mixin default is 512).
    max_tokens = 8192
    # Client-side eval concurrency (mixin default is 1).
    num_threads = 4
    env = {"SGLANG_USE_SGL_XPU": "1"}

    # Hybrid-mamba layout needs --model-impl sglang, a fixed page size, and
    # the nemotron_3 reasoning / qwen3_coder tool-call parsers.
    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--max-total-tokens",
        "65536",
        "--mem-fraction-static",
        "0.85",
        "--context-length",
        "16384",
        "--page-size",
        "64",
        "--chunked-prefill-size",
        "1024",
        "--max-running-requests",
        "8",
        "--watchdog-timeout",
        "1200",
        "--model-impl",
        "sglang",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "nemotron_3",
    ]


if __name__ == "__main__":
    unittest.main()
