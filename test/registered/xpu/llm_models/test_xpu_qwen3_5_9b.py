"""Qwen3.5-9B GSM8K accuracy on Intel XPU (TP=4).

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
class TestQwen3_5_9BXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "Qwen/Qwen3.5-9B"
    tp_size = 4
    accuracy = 0.90
    # The earlier num_threads=4 accuracy drop (0.245 vs 0.60) was an artifact
    # of the default max_tokens=512 truncating GSM8K CoT mid-answer. At
    # max_tokens=8192 the CoT completes and concurrency no longer hurts:
    # tp=4, num_threads=4, max_tokens=8192, gsm8k n=50 scored 0.96 (reproduced
    # across two runs), at ~98 tok/s and ~29 min wall clock. num_threads=4
    # sharing 4 tiles cleared the Level Zero wedge that single-stream tp=1 was
    # working around, so the 4-gpu suite is the right home for this test.
    num_threads = 4
    max_tokens = 8192

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
