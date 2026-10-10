"""
Disaggregation integration tests for the Mooncake transfer backend on Intel XPU.

Launches a prefill server, a decode server, and a load-balancer, then verifies
that basic text completion works end-to-end. Mooncake runs on the TENT engine
(MC_USE_TENT=1), whose XPU platform stages device memory through host DRAM,
over the TCP transport so no RDMA NIC is needed.

Requirements:
    The ``sglang-router`` package and a mooncake-transfer-engine built with
    -DUSE_XPU=ON -DUSE_TENT=ON (both installed by docker/xpu.Dockerfile). The
    NIXL variant is test/manual/kv_transfer/test_disaggregation_nixl_xpu.py.

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

register_xpu_ci(est_time=300, suite="nightly-xpu-kernel-main-2-gpu", nightly=True)

_XPU_AVAILABLE = torch.xpu.is_available()
# PD disaggregation needs two devices: the fixture puts decode on
# decode_base_gpu_id=1 while prefill holds device 0.
_XPU_DEVICE_COUNT = torch.xpu.device_count() if _XPU_AVAILABLE else 0
_SKIP_REASON = (
    "Intel XPU not available (torch.xpu.is_available() returned False)"
    if not _XPU_AVAILABLE
    else f"PD disaggregation needs 2 XPUs, found {_XPU_DEVICE_COUNT}"
)

# The XPU CI image gets Mooncake from docker/xpu.Dockerfile; skip, not fail,
# on an image built before that step landed.
try:
    import mooncake.engine  # noqa: F401

    _MOONCAKE_AVAILABLE = True
except ImportError:
    _MOONCAKE_AVAILABLE = False
_MOONCAKE_SKIP = "mooncake transfer engine not installed"

# XPU support lives in Mooncake's TENT engine. With no RDMA NIC, TCP carries
# the host-staged bytes between prefill and decode.
_MOONCAKE_XPU_ENV = {
    "MC_USE_TENT": "1",
    "MOONCAKE_PROTOCOL": "tcp",
}


class _DisaggregationXpuTestMixin:
    """Backend-agnostic completion checks; subclasses pick the backend."""

    transfer_backend_name: str

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.transfer_backend = [
            "--disaggregation-transfer-backend",
            cls.transfer_backend_name,
        ]
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
        """Disaggregated output must produce the expected token for a deterministic prompt."""
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


@unittest.skipUnless(_MOONCAKE_AVAILABLE, _MOONCAKE_SKIP)
@unittest.skipUnless(_XPU_DEVICE_COUNT >= 2, _SKIP_REASON)
class TestDisaggregationMooncakeBasic(
    _DisaggregationXpuTestMixin, PDDisaggregationServerBase
):
    """Smoke-test the Mooncake disaggregation backend with a small completion."""

    transfer_backend_name = "mooncake"
    extra_prefill_env = _MOONCAKE_XPU_ENV
    extra_decode_env = _MOONCAKE_XPU_ENV


if __name__ == "__main__":
    unittest.main()
