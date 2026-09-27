"""
Disaggregation integration tests for the NIXL and Mooncake backends with the
staging buffer on Intel XPU.

Tests the staging buffer optimization for KV cache transfer in PD disaggregation.
The staging buffer reduces RDMA request count from O(tokens * layers) to O(1)
by gathering scattered KV head slices into contiguous GPU memory before bulk transfer.

Requirements:
    The ``sglang-router`` package and a mooncake-transfer-engine built with
    -DUSE_XPU=ON -DUSE_TENT=ON (both installed by docker/xpu.Dockerfile). The
    NIXL classes are skipped until NIXL and UCX support XPU memory.
"""

import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

register_xpu_ci(est_time=300, suite="nightly-xpu-4-gpu", nightly=True)

_XPU_AVAILABLE = torch.xpu.is_available()
# Prefill holds device 0 and decode TP 2 takes devices 1-2. Both backends only
# stage when decode TP differs from prefill attn TP; equal TP sends pages
# directly.
_XPU_DEVICE_COUNT = torch.xpu.device_count() if _XPU_AVAILABLE else 0
_SKIP_REASON = (
    "Intel XPU not available (torch.xpu.is_available() returned False)"
    if not _XPU_AVAILABLE
    else f"Heterogeneous-TP PD needs 3 XPUs, found {_XPU_DEVICE_COUNT}"
)

# UCX treats SYCL device allocations as host memory, so NIXL cannot register
# XPU KV buffers. Drop once both upstream PRs are in the XPU CI image.
_NIXL_XPU_SKIP = (
    "NIXL on XPU needs https://github.com/ai-dynamo/nixl/pull/1536 and "
    "https://github.com/openucx/ucx/pull/11218"
)

# The XPU CI image gets Mooncake from docker/xpu.Dockerfile; skip, not fail,
# on an image built before that step landed.
try:
    import mooncake.engine  # noqa: F401

    _MOONCAKE_AVAILABLE = True
except ImportError:
    _MOONCAKE_AVAILABLE = False
_MOONCAKE_SKIP = "mooncake transfer engine not installed"


_STAGING_PREFILL_ENV = {"SGLANG_DISAGG_STAGING_BUFFER": "1"}
_STAGING_DECODE_ENV = {
    "SGLANG_DISAGG_STAGING_BUFFER": "1",
    "SGLANG_DISAGG_STAGING_POOL_SIZE_MB": "512",
    # At TP > 1 the torch.compile'd vocab-parallel embedding mask segfaults in
    # inductor's static Triton launcher on XPU; run it eager.
    "TORCHDYNAMO_DISABLE": "1",
}
# XPU support lives in Mooncake's TENT engine. With no RDMA NIC, TCP carries
# the host-staged bytes between prefill and decode.
_MOONCAKE_XPU_ENV = {
    "MC_USE_TENT": "1",
    "MOONCAKE_PROTOCOL": "tcp",
}
_NIXL_XPU_ENV = {"UCX_TLS": "ze_copy,ze_ipc,ib,tcp"}


class _DisaggregationStagingXpuTestMixin:
    """Backend-agnostic staging checks; subclasses pick the backend."""

    decode_tp_size = 2
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

    def test_completion_works_with_staging(self):
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

    def test_completion_deterministic_output(self):
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

    def _generate(self, prompt: str) -> str:
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 50},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["text"]

    def test_concurrent_matches_serial_with_staging(self):
        """Concurrent scatter must not corrupt KV that serial scatter gets right.

        Under load the ring buffer recycles slots and several rooms scatter at
        once; a slot freed before its scatter drains, or a watermark that lets
        prefill overwrite un-scattered bytes, changes the decoded tokens. At
        temperature 0 the serial answers are the reference, so any such
        divergence shows up as a mismatch rather than as merely odd text.
        """
        import concurrent.futures

        # Distinct lengths and topics, so a mis-scattered region lands on
        # different KV pages per prompt instead of cancelling out.
        prompts = [
            "The capital of France is",
            "Explain in two sentences what a compiler does.",
            "List the first six prime numbers in order, separated by commas.",
        ]

        serial = [self._generate(p) for p in prompts]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(prompts)
        ) as executor:
            concurrent_out = list(executor.map(self._generate, prompts))

        for prompt, expected, got in zip(prompts, serial, concurrent_out):
            self.assertEqual(
                got,
                expected,
                f"Concurrent staging output diverged from serial for {prompt!r}",
            )

    def test_long_sequence_generation(self):
        """A multi-chunk generation must complete with staging enabled."""
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "Write a detailed explanation of how computers work, "
                "covering CPUs, memory, storage, and networking. ",
                "sampling_params": {"temperature": 0, "max_new_tokens": 200},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        self.assertIn("text", data)
        self.assertGreater(len(data["text"]), 100, "Long sequence should generate text")

        self.assertIn("meta_info", data)
        meta = data["meta_info"]
        self.assertIn("completion_tokens", meta)
        self.assertGreater(
            meta["completion_tokens"],
            50,
            "Should generate substantial number of tokens",
        )


@unittest.skip(_NIXL_XPU_SKIP)
@unittest.skipUnless(_XPU_DEVICE_COUNT >= 3, _SKIP_REASON)
class TestDisaggregationNixlStaging(
    _DisaggregationStagingXpuTestMixin, PDDisaggregationServerBase
):
    transfer_backend_name = "nixl"
    extra_prefill_env = {**_STAGING_PREFILL_ENV, **_NIXL_XPU_ENV}
    extra_decode_env = {**_STAGING_DECODE_ENV, **_NIXL_XPU_ENV}


@unittest.skipUnless(_MOONCAKE_AVAILABLE, _MOONCAKE_SKIP)
@unittest.skipUnless(_XPU_DEVICE_COUNT >= 3, _SKIP_REASON)
class TestDisaggregationMooncakeStaging(
    _DisaggregationStagingXpuTestMixin, PDDisaggregationServerBase
):
    transfer_backend_name = "mooncake"
    extra_prefill_env = {**_STAGING_PREFILL_ENV, **_MOONCAKE_XPU_ENV}
    extra_decode_env = {**_STAGING_DECODE_ENV, **_MOONCAKE_XPU_ENV}


if __name__ == "__main__":
    unittest.main()
