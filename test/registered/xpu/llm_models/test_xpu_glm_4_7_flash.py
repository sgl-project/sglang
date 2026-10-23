"""GLM-4.7-Flash GSM8K accuracy on Intel XPU (TP=4).

Scored by ``simple_eval_gsm8k.GSM8KEval``.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=2400, suite="nightly-xpu-kernel-main-4-gpu", nightly=True)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestGLM4_7FlashXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "zai-org/GLM-4.7-Flash"
    tp_size = 4
    accuracy = 0.90
    timeout_for_server_launch = 3600
    num_examples = 50
    num_threads = 4
    max_tokens = 8192

    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--mem-fraction-static",
        "0.85",
        # More concurrent requests lose the device on Arc Pro B60.
        "--max-running-requests",
        "4",
    ]


if __name__ == "__main__":
    unittest.main()
