"""Qwen3.5-9B GSM8K accuracy on Intel XPU (TP=1).

Scored by sgl-eval ``gsm8k`` with thinking disabled. Covers both standalone SYCL GDN
(``--linear-attn-backend intel_xpu``) and Triton GDN. Both use SYCL conv1d.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=900, suite="nightly-xpu-kernel-main-4-gpu", nightly=True)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class Qwen3_5_9BXPUBase(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "Qwen/Qwen3.5-9B"
    tp_size = 1
    # Thinking loops re-checking sgl-eval's boxed answer until max_tokens;
    # non-thinking scored 0.96 on 50 examples (3 runs, B70).
    chat_template_kwargs = {"enable_thinking": False}
    # Lowest score over 3 runs of all 3 classes (B70); raise only on new data.
    accuracy = 0.95
    num_examples = 20
    num_threads = 4
    max_tokens = 2048

    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--page-size",
        "128",
        "--max-total-tokens",
        "65536",
        "--mem-fraction-static",
        "0.85",
    ]


class TestQwen3_5_9BXPUDefault(Qwen3_5_9BXPUBase):
    """Default Triton GDN recurrence with SYCL conv1d."""


class TestQwen3_5_9BXPUStandaloneGDN(Qwen3_5_9BXPUBase):
    """Standalone SYCL GDN recurrence (``--linear-attn-backend intel_xpu``)."""

    other_args = Qwen3_5_9BXPUBase.other_args + [
        "--linear-attn-backend",
        "intel_xpu",
    ]


if __name__ == "__main__":
    unittest.main()
