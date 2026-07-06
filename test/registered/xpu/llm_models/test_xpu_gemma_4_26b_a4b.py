"""gemma-4-26B-A4B GSM8K accuracy on Intel XPU (TP=4).

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
class TestGemma4_26BA4BXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "google/gemma-4-26B-A4B-it"
    tp_size = 4
    accuracy = 0.90
    timeout_for_server_launch = 3600
    env = {"SGLANG_USE_SGL_XPU": "1"}

    # Server args mirror /data/pgirijal/scripts/run_upstream_key_models.sh
    # accuracy_commands["google/gemma-4-26B-A4B-it"]. Gemma-4 hybrid-attention
    # kernels crash under chunked prefill on Intel XPU; keep --chunked-prefill-size -1.
    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--page-size",
        "64",
        "--max-total-tokens",
        "65536",
        "--mem-fraction-static",
        "0.9",
        "--chunked-prefill-size",
        "-1",
    ]


if __name__ == "__main__":
    unittest.main()
