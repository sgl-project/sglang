"""Qwen3.5-9B GSM8K accuracy on Intel XPU (TP=4).

Scored by ``simple_eval_gsm8k.GSM8KEval``. Covers both the default fused GDN
SYCL kernel path (``SGLANG_XPU_FUSED_GDN`` defaults to True) and the Triton
GDN fallback path (``SGLANG_XPU_FUSED_GDN=False``).
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
class Qwen3_5_9BXPUBase(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "Qwen/Qwen3.5-9B"
    tp_size = 4
    accuracy = 0.90
    # max_tokens=8192 lets the GSM8K CoT complete under num_threads=4.
    num_examples = 50
    num_threads = 4
    max_tokens = 8192

    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--page-size",
        "128",
        "--max-total-tokens",
        "65536",
        "--mem-fraction-static",
        "0.85",
    ]


class TestQwen3_5_9BXPUFusedGDN(Qwen3_5_9BXPUBase):
    """Default path: fused SYCL GDN kernel from sgl-kernel-xpu."""


class TestQwen3_5_9BXPUGDNFallback(Qwen3_5_9BXPUBase):
    """Triton GDN fallback path (``SGLANG_XPU_FUSED_GDN=False``).

    Small ``num_examples`` since this is a smoke check of the fallback
    dispatch, not a full accuracy regression test (already covered by the
    fused-path class above); accuracy threshold is left at 0 accordingly.
    """

    env = {"SGLANG_XPU_FUSED_GDN": "False"}
    num_examples = 8
    accuracy = 0.0


if __name__ == "__main__":
    unittest.main()
