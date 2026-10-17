"""
Disaggregation integration tests for the NIXL backend with the staging buffer on
Intel XPU.

Manual, not CI: UCX treats SYCL device allocations as host memory, so NIXL
cannot register XPU KV buffers until https://github.com/ai-dynamo/nixl/pull/1536
and https://github.com/openucx/ucx/pull/11218 land. Run against NIXL and UCX
builds that carry both patches:

    python3 test/manual/kv_transfer/test_disaggregation_staging_nixl_xpu.py

Mirrors the Mooncake scenarios in
test/registered/xpu/test_disaggregation_staging_xpu.py. Needs 3 XPUs.
"""

import concurrent.futures
import unittest

import requests
import torch

from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

_XPU_AVAILABLE = torch.xpu.is_available()
# Prefill holds device 0 and decode TP 2 takes devices 1-2. NIXL only stages
# when decode TP differs from prefill attn TP; equal TP sends pages directly.
_XPU_DEVICE_COUNT = torch.xpu.device_count() if _XPU_AVAILABLE else 0
_SKIP_REASON = (
    "Intel XPU not available (torch.xpu.is_available() returned False)"
    if not _XPU_AVAILABLE
    else f"Heterogeneous-TP PD needs 3 XPUs, found {_XPU_DEVICE_COUNT}"
)

try:
    import nixl._api  # noqa: F401

    _NIXL_AVAILABLE = True
except ImportError:
    _NIXL_AVAILABLE = False
_NIXL_SKIP = "nixl not installed"

_NIXL_XPU_ENV = {"UCX_TLS": "ze_copy,ze_ipc,ib,tcp"}
_STAGING_PREFILL_ENV = {"SGLANG_DISAGG_STAGING_BUFFER": "1", **_NIXL_XPU_ENV}
_STAGING_DECODE_ENV = {
    "SGLANG_DISAGG_STAGING_BUFFER": "1",
    "SGLANG_DISAGG_STAGING_POOL_SIZE_MB": "512",
    # Workaround: on torch 2.13+xpu inductor's static Triton launcher segfaults in
    # load_kernel on the TP-sharded embedding mask, PD or not; likely
    # pytorch/pytorch#199511, drop once the XPU torch pin includes that fix.
    "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER": "0",
    **_NIXL_XPU_ENV,
}


@unittest.skipUnless(_NIXL_AVAILABLE, _NIXL_SKIP)
@unittest.skipUnless(_XPU_DEVICE_COUNT >= 3, _SKIP_REASON)
class TestDisaggregationNixlStaging(PDDisaggregationServerBase):
    decode_tp_size = 2
    extra_prefill_env = _STAGING_PREFILL_ENV
    extra_decode_env = _STAGING_DECODE_ENV

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

    def _generate(self, prompt: str, max_new_tokens: int = 50) -> dict:
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_completion_works_with_staging(self):
        data = self._generate("The capital of France is", max_new_tokens=16)
        self.assertGreater(len(data["text"]), 0, "Generated text should not be empty")

    def test_completion_deterministic_output(self):
        generated = self._generate("1 + 1 =", max_new_tokens=4)["text"]
        self.assertIn("2", generated, f"Expected '2' in output, got: {generated!r}")

    def test_concurrent_matches_serial_with_staging(self):
        """Concurrent scatter must not corrupt KV that serial scatter gets right."""
        # Distinct lengths and topics, so a mis-scattered region lands on
        # different KV pages per prompt instead of cancelling out.
        prompts = [
            "The capital of France is",
            "Explain in two sentences what a compiler does.",
            "List the first six prime numbers in order, separated by commas.",
        ]
        serial = [self._generate(p)["text"] for p in prompts]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(prompts)
        ) as executor:
            concurrent_out = [d["text"] for d in executor.map(self._generate, prompts)]

        for prompt, expected, got in zip(prompts, serial, concurrent_out):
            self.assertEqual(
                got,
                expected,
                f"Concurrent staging output diverged from serial for {prompt!r}",
            )

    def test_long_sequence_generation(self):
        """A multi-chunk generation must complete with staging enabled."""
        data = self._generate(
            "Write a detailed explanation of how computers work, "
            "covering CPUs, memory, storage, and networking. ",
            max_new_tokens=200,
        )
        self.assertGreater(len(data["text"]), 100, "Long sequence should generate text")
        self.assertGreater(
            data["meta_info"]["completion_tokens"],
            50,
            "Should generate substantial number of tokens",
        )


if __name__ == "__main__":
    unittest.main()
