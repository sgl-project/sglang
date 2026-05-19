"""End-to-end canary test on DeepSeek-V4-Flash (1-GPU, 5-layer override).

DSV4-Flash is the most complex hybrid pool sglang supports (43 layers,
compress_ratios mixing full/4x/128x, 5 internal sub-pools). The point
of this test is to prove the canary handles it out-of-the-box by always
attaching a standard full + a standard swa canary buffer on the top-level
``DeepSeekV4TokenToKVPool`` without diving into c4 / c128 / indexer /
compress_state internals.

Single-GPU constraint:

- ``--load-format dummy`` is unusable on DSV4 (init paths read real
  weights for the expert / hash layer / compressor setup), so we load
  real weights but override ``num_hidden_layers`` to 5.
- HF config's ``compress_ratios`` length must match
  ``num_hidden_layers``; using the first-5 prefix ``[0, 0, 4, 128, 4]``
  exercises all three compression flavours (1 full / 2 c4 / 1 c128).
- DSV4-Flash ships with FP4-packed MoE expert weights. The default
  fused_experts_impl path expects bf16 / fp8 layouts and trips on
  ``Hidden size mismatch`` in ``fused_moe.py:828``; ``--moe-runner-backend
  marlin`` routes experts through the Marlin kernel which handles the
  FP4 padding correctly.
"""

from __future__ import annotations

import time
import unittest

from sglang.test.canary_e2e_base import CanaryE2EBase
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=420, stage="extra-b", runner_config="8-gpu-h200")

_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
# First-5 prefix of the real HF ``compress_ratios`` array; covers full
# (ratio=0), c4 (ratio=4), and c128 (ratio=128) so all three internal
# sub-pools get exercised even with the truncated layer count.
_OVERRIDE_ARGS = '{"num_hidden_layers": 5, "compress_ratios": [0, 0, 4, 128, 4]}'


class TestKvCacheCanaryDSV4Flash(CanaryE2EBase):
    """Launch DSV4-Flash (5 layers, real weights) with canary in raise mode."""

    model = _MODEL
    extra_server_args = [
        "--trust-remote-code",
        "--tp",
        "1",
        "--json-model-override-args",
        _OVERRIDE_ARGS,
        "--moe-runner-backend",
        "marlin",
        "--watchdog-timeout",
        "900",
    ]

    def test_clean_run_no_canary_violation(self) -> None:
        results = self.send_parallel_requests(n=16, max_new_tokens=16, timeout=120.0)
        bad = [r for r in results if r.get("status_code") != 200]
        self.assertFalse(bad, f"non-200 responses on DSV4 clean run: {bad[:3]}")

        # Allow the side-stream event pump a beat to refresh canary
        # counters and let any pending allreduce settle.
        time.sleep(2.0)

        self.assert_health_ok()


if __name__ == "__main__":
    unittest.main(verbosity=3)
