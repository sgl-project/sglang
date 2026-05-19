"""End-to-end canary test on DeepSeek-V4-Flash (1-GPU, 5-layer override).

DSV4-Flash is the most complex hybrid pool sglang supports (43 layers,
compress_ratios mixing full/4x/128x, 5 internal sub-pools). The point
of this test is to prove the canary handles it out-of-the-box by the
"full + swa 两条都走" rule -- the canary attaches a standard full + a
standard swa shadow on the top-level ``DeepSeekV4TokenToKVPool``
without diving into c4 / c128 / indexer / compress_state internals.

Single-GPU constraint:

- ``--load-format dummy`` is unusable on DSV4 (init paths read real
  weights for the expert / hash layer / compressor setup), so we load
  real weights but override ``num_hidden_layers`` to 5.
- HF config's ``compress_ratios`` length must match ``num_hidden_layers``;
  using the first-5 prefix ``[0, 0, 4, 128, 4]`` exercises all three
  compression flavours (1 full / 2 c4 / 1 c128).
- DSV4-Flash ships with FP4-packed MoE expert weights. The default
  fused_experts_impl path expects bf16 / fp8 layouts and trips on
  ``Hidden size mismatch`` in ``fused_moe.py:828``; ``--moe-runner-backend
  marlin`` routes experts through the Marlin kernel which handles the
  FP4 padding correctly (matches the upstream LowLatency FP4 recipe at
  test/registered/dsv4/test_deepseek_v4_flash_fp4_h200.py).

Test group: extra-a / 1-gpu-large (H100 80GB equivalent). Not base --
launching DSV4 takes several minutes which is too heavy for the
per-PR base lane.
"""

from __future__ import annotations

import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=420, stage="extra-b", runner_config="8-gpu-h200")

_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
# First-5 prefix of the real HF ``compress_ratios`` array; covers full
# (ratio=0), c4 (ratio=4), and c128 (ratio=128) so all three internal
# sub-pools get exercised even with the truncated layer count.
_OVERRIDE_ARGS = '{"num_hidden_layers": 5, "compress_ratios": [0, 0, 4, 128, 4]}'


class TestKvCacheCanaryDSV4Flash(CustomTestCase):
    """Launch DSV4-Flash (5 layers, real weights) with canary in raise mode."""

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "1",
                "--json-model-override-args",
                _OVERRIDE_ARGS,
                "--moe-runner-backend",
                "marlin",
                "--watchdog-timeout",
                "900",
                "--kv-cache-canary",
                "raise",
                "--mem-fraction-static",
                "0.65",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_clean_run_no_canary_violation(self):
        # Step 1: send a small batch of generate requests across a few
        # different prompt patterns to exercise both prefill and decode.
        prompts = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "Explain in one sentence what a transformer is.",
            "1 + 1 =",
        ]
        for i, prompt in enumerate(prompts * 4):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": f"{prompt} {i}",
                    "sampling_params": {"max_new_tokens": 16, "temperature": 0.0},
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200, response.text)

        # Step 2: allow the side-stream event pump a beat to refresh
        # canary counters and let any pending allreduce settle.
        time.sleep(2.0)

        # Step 3: server must still be healthy. In raise mode any
        # canary violation aborts the server, so a successful health
        # check is the test signal.
        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=3)
