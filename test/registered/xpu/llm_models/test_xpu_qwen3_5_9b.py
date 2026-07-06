"""Qwen3.5-9B GSM8K accuracy on Intel XPU (TP=1).

Scored by ``simple_eval_gsm8k.GSM8KEval`` (the same evaluator AMD and
NVIDIA nightlies use).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=4000, suite="nightly-xpu-1-gpu", nightly=True)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestQwen3_5_9BXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "Qwen/Qwen3.5-9B"
    tp_size = 1
    accuracy = 0.55
    # 200 questions @ ~21 s/q single-thread = ~70 min; the runner's
    # per-file budget must cover this (see the 4200s --timeout-per-file
    # in .github/workflows/nightly-test-intel.yml). num_threads=4 halved
    # wall clock but hurt GSM8K CoT accuracy (0.245 vs 0.60 baseline);
    # single-thread is required to match the reference score.

    # Server args mirror /data/pgirijal/scripts/run_upstream_key_models.sh
    # accuracy_commands["Qwen/Qwen3.5-9B"]. --disable-radix-cache /
    # --disable-overlap-schedule / --dtype bfloat16 / --trust-remote-code /
    # --attention-backend intel_xpu / --device xpu come from the mixin base.
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
