"""
Disaggregation integration tests for the Mooncake and NIXL transfer backends on Intel XPU.

Launches a prefill server, a decode server, and a load-balancer, then verifies
that basic text completion works end-to-end. Mooncake runs on the TENT engine
(MC_USE_TENT=1), which copies KV card to card through Level Zero IPC on one
node, so no RDMA NIC is needed. NIXL runs on UCX with the
Level Zero transport; device addresses there have bit 63 set, which exercises
the np.uint64 pointer arithmetic in disaggregation/nixl/conn.py.

Requirements:
    The ``sglang-router`` package, a mooncake-transfer-engine built with
    -DUSE_XPU=ON -DUSE_TENT=ON, and NIXL on a UCX built --with-ze (all installed
    by docker/xpu.Dockerfile).

Usage:
    python3 -m pytest test/registered/disaggregation/test_disaggregation_xpu.py -v
"""

import importlib
import time
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN, is_in_ci

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
# Outside CI, skip when fewer than 2 XPUs are visible; in CI the class runs and
# setUpClass fails, so a runner with a missing card is not silently green.
_requires_two_xpus = unittest.skipUnless(
    _XPU_DEVICE_COUNT >= 2 or is_in_ci(), _SKIP_REASON
)

# XPU support lives in Mooncake's TENT engine; with MOONCAKE_PROTOCOL left at
# its default it copies KV card to card through Level Zero IPC.
_MOONCAKE_XPU_ENV = {"MC_USE_TENT": "1"}


class _DisaggregationXpuTestMixin:
    """Backend-agnostic completion checks; subclasses pick the backend."""

    transfer_backend_name: str
    backend_module: str
    # The XPU runners have no RDMA NIC; in CI the fixture would otherwise pass its
    # CUDA-runner default (mlx5_roce*), which Mooncake rejects at startup.
    use_rdma = False

    @classmethod
    def setUpClass(cls):
        if _XPU_DEVICE_COUNT < 2:
            raise RuntimeError(_SKIP_REASON)
        # Fail rather than skip, so a CI image missing a backend build from
        # docker/xpu.Dockerfile shows up red instead of silently green.
        importlib.import_module(cls.backend_module)
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.transfer_backend = [
            "--disaggregation-transfer-backend",
            cls.transfer_backend_name,
        ]
        # intel_xpu is the documented XPU attention backend (page size 128). A small
        # prefill chunk splits the long-prompt test's KV into several transfers.
        xpu_args = ["--device", "xpu", "--attention-backend", "intel_xpu"]
        cls.extra_prefill_args = xpu_args + ["--chunked-prefill-size", "256"]
        cls.extra_decode_args = xpu_args
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

    def test_long_prompt_multi_page_transfer(self):
        """A fact at the start of a long prompt must survive the KV transfer.

        The prompt spans several 128-token KV pages and prefill chunks, and the
        fact lands in the first page, so a dropped or misplaced page changes the
        greedy answer.
        """
        filler = "The sky was clear and the market was quiet that day. " * 80
        prompt = (
            "Remember this: the secret code is 7342.\n"
            + filler
            + "\nQuestion: What is the secret code? Answer:"
        )
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        generated = response.json()["text"]
        self.assertIn("7342", generated, f"Expected '7342', got: {generated!r}")


@_requires_two_xpus
class TestDisaggregationMooncakeBasic(
    _DisaggregationXpuTestMixin, PDDisaggregationServerBase
):
    """Smoke-test the Mooncake disaggregation backend with a small completion."""

    transfer_backend_name = "mooncake"
    backend_module = "mooncake.engine"
    extra_prefill_env = _MOONCAKE_XPU_ENV
    extra_decode_env = _MOONCAKE_XPU_ENV
    # TENT falls back to another route when it cannot map the peer's KV buffer
    # (for example without --cap-add SYS_PTRACE), and requests still succeed, so
    # check the prefill log for the card-to-card path.
    capture_per_side_logs = True

    def test_kv_copied_card_to_card(self):
        """Prefill must map decode's KV buffers through Level Zero IPC.

        Needs a Mooncake build with kvcache-ai/Mooncake#4379 (as in
        docker/xpu.Dockerfile); an older XPU build copies through the host and
        fails this test while the other Mooncake tests still pass.
        """
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        # The fixture's log reader threads run behind the response.
        deadline = time.monotonic() + 10
        while (
            "XpuTransport: mapped peer buffer"
            not in (log := self._prefill_stderr_buf.getvalue())
            and time.monotonic() < deadline
        ):
            time.sleep(0.5)
        self.assertIn("XpuTransport: mapped peer buffer", log)
        self.assertNotIn("using another route", log)


@_requires_two_xpus
class TestDisaggregationNixlBasic(
    _DisaggregationXpuTestMixin, PDDisaggregationServerBase
):
    """Smoke-test the NIXL disaggregation backend with a small completion."""

    transfer_backend_name = "nixl"
    backend_module = "nixl._api"


if __name__ == "__main__":
    unittest.main()
