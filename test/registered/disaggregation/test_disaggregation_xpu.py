"""
Disaggregation integration test for the NIXL transfer backend on Intel XPU.

Launches a prefill server, a decode server, and a load-balancer using the
NIXL KV-transfer backend, then verifies that basic text completion works
end-to-end.  This exercises the np.uint64 pointer-arithmetic fix in
python/sglang/srt/disaggregation/nixl/conn.py, which is required on
Intel XPU where device addresses have bit 63 set (e.g. 0xffff81ab54e01000)
and would overflow np.int64.

Requirements:
    The ``sglang-router`` package must be installed in the environment (it is
    provided by the XPU/disagg test image, same as other PD tests).

Usage:
    python3 -m pytest test/registered/disaggregation/test_disaggregation_xpu.py -v
"""

import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

register_xpu_ci(
    est_time=300,
    suite="nightly-xpu-2-gpu",
    nightly=True,
    disabled=(
        "Requires NIXL XPU support which depends on upstream PRs not yet merged: "
        "NIXL https://github.com/ai-dynamo/nixl/pull/1536 and "
        "UCX https://github.com/openucx/ucx/pull/11218"
    ),
)

_XPU_AVAILABLE = torch.xpu.is_available()
# PD disaggregation needs two devices: the fixture puts decode on
# decode_base_gpu_id=1 while prefill holds device 0.
_XPU_DEVICE_COUNT = torch.xpu.device_count() if _XPU_AVAILABLE else 0
_SKIP_REASON = (
    "Intel XPU not available (torch.xpu.is_available() returned False)"
    if not _XPU_AVAILABLE
    else f"PD disaggregation needs 2 XPUs, found {_XPU_DEVICE_COUNT}"
)


@unittest.skipUnless(_XPU_DEVICE_COUNT >= 2, _SKIP_REASON)
class TestDisaggregationNixlBasic(PDDisaggregationServerBase):
    """Smoke-test the NIXL disaggregation backend with a small completion."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.rdma_devices = []
        cls.extra_prefill_args = ["--device", "xpu"]
        # host_pool retraction backup calls cudaHostRegister, which is CUDA-only.
        cls.extra_decode_args = [
            "--device",
            "xpu",
            "--disaggregation-decode-retraction-backup",
            "cpu_tensor",
        ]
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
        self.assertIn("2", generated, f"Expected '2' in output, got: {generated!r}")


if __name__ == "__main__":
    unittest.main()
