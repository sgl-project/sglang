"""Qwen3.5-35B-A3B GSM8K accuracy on Intel XPU (TP=4).

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
class TestQwen3_5_35BA3BXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "Qwen/Qwen3.5-35B-A3B"
    tp_size = 4
    accuracy = 0.60
    # MoE weights split across 4 ranks take several minutes to load on Intel
    # Arc Pro B60; mirror the 32B/70B XPU tests' 1-hour launch budget.
    timeout_for_server_launch = 3600
    # SGL XPU MoE kernels gate on this env var (see launch command in the
    # Intel XPU nightly runbook).
    env = {"SGLANG_USE_SGL_XPU": "1"}

    # Server args mirror /data/pgirijal/scripts/run_upstream_key_models.sh
    # accuracy_commands["Qwen/Qwen3.5-35B-A3B"].
    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--page-size",
        "128",
        "--max-total-tokens",
        "65536",
        "--mem-fraction-static",
        "0.85",
    ]


if __name__ == "__main__":
    unittest.main()
