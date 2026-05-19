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

import os
import time
import unittest
from typing import ClassVar, Dict, List, Optional

import requests

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


class TestKvCacheCanaryDSV4FlashPerturbed(TestKvCacheCanaryDSV4Flash):
    """DSV4-Flash + real-KV byte perturb: per-step head/tail must raise.

    Inherits model + extra_server_args from the clean sibling. The perturb
    hook flips one real-KV byte; with no sweep configured here, the
    per-step ``head`` and ``tail`` kernels are what observe the
    corruption. Either the warmup forward trips canary (server fails to
    come up) or the live burst returns errors / /health flips.
    """

    server_env: ClassVar[Dict[str, str]] = {
        "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB": "0.01",
        "SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED": "42",
    }
    allow_launch_failure = True

    _saved_env: ClassVar[Optional[Dict[str, Optional[str]]]] = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._saved_env = {k: os.environ.get(k) for k in cls.server_env}
        for k, v in cls.server_env.items():
            os.environ[k] = v
        try:
            super().setUpClass()
        except Exception:
            cls._restore_env()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            super().tearDownClass()
        finally:
            cls._restore_env()

    @classmethod
    def _restore_env(cls) -> None:
        if cls._saved_env is None:
            return
        for k, old in cls._saved_env.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        cls._saved_env = None

    def test_perturbed_burst_triggers_canary_raise(self) -> None:
        if not self.launch_failed:
            results: List[Dict[str, object]] = self.send_parallel_requests(
                n=200, max_new_tokens=32, timeout=30.0
            )
            triggered = any(
                "error" in r or int(r.get("status_code", 0)) >= 500 for r in results
            )

            time.sleep(1.5)
            try:
                health_ok = (
                    requests.get(self.base_url + "/health", timeout=5).status_code
                    == 200
                )
            except requests.exceptions.RequestException:
                health_ok = False

            self.assertTrue(
                triggered or not health_ok,
                f"Expected canary to fire under perturb+raise, but server still "
                f"healthy and no failed requests; first 3 responses: {results[:3]}",
            )
        # Without sweep enabled, the per-step head/tail kernels are the
        # only observers of the byte flip on DSV4 Flash.
        self.assert_violation_kind_logged(["head_k", "head_v", "tail_k", "tail_v"])


if __name__ == "__main__":
    unittest.main(verbosity=3)
