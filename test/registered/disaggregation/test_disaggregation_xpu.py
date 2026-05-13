"""
Disaggregation integration test for the NIXL transfer backend on Intel XPU.

Launches a prefill server, a decode server, and a load-balancer using the
NIXL KV-transfer backend, then verifies that basic text completion works
end-to-end.  This exercises the np.uint64 pointer-arithmetic fix in
python/sglang/srt/disaggregation/nixl/conn.py, which is required on
Intel XPU where device addresses have bit 63 set (e.g. 0xffff81ab54e01000)
and would overflow np.int64.

Usage:
    python3 -m pytest test/registered/disaggregation/test_disaggregation_xpu.py -v
"""

import subprocess
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

register_cuda_ci(
    est_time=300,
    suite="stage-a-test-1-gpu-small",
    disabled="Intel XPU only — not available in standard CUDA CI",
)

_XPU_AVAILABLE = torch.xpu.is_available()


@unittest.skipUnless(
    _XPU_AVAILABLE, "Intel XPU not available (torch.xpu.is_available() returned False)"
)
class TestDisaggregationNixlBasic(PDDisaggregationServerBase):
    """Smoke-test the NIXL disaggregation backend with a small completion."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        # Force the NIXL backend and XPU device.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.rdma_devices = []
        cls.extra_prefill_args = ["--device", "xpu"]
        cls.extra_decode_args = ["--device", "xpu"]
        subprocess.check_call(
            ["pip", "install", "sglang-router"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cls.launch_all()

    def test_completion_returns_text(self):
        """A simple completion must succeed and return non-empty generated text."""
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIn("text", data, f"Unexpected response shape: {data}")
        self.assertGreater(
            len(data["text"]),
            0,
            "Generated text should not be empty",
        )

    def test_completion_correct_output(self):
        """Disaggregated NIXL output must produce the expected token for a deterministic prompt."""
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "1 + 1 =",
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        generated = response.json()["text"]
        # The model should produce "2" somewhere in the first few tokens.
        self.assertIn("2", generated, f"Expected '2' in output, got: {generated!r}")


if __name__ == "__main__":
    unittest.main()
