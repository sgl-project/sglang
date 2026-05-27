"""
Disaggregation integration test for NIXL backend with staging buffer on Intel XPU.

Tests the staging buffer optimization for KV cache transfer in PD disaggregation.
The staging buffer reduces RDMA request count from O(tokens * layers) to O(1)
by gathering scattered KV head slices into contiguous GPU memory before bulk transfer.

This test verifies:
1. Staging buffer allocation on both prefill and decode sides
2. End-to-end text completion works with staging enabled
3. Staging buffers are properly registered with NIXL

Requirements:
    The ``sglang-router`` package must be installed in the environment (it is
    provided by the XPU/disagg test image, same as other PD tests).

Usage:
    python3 -m pytest test/registered/disaggregation/test_disaggregation_xpu_staging.py -v
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
class TestDisaggregationNixlStaging(PDDisaggregationServerBase):
    """Test NIXL disaggregation backend with staging buffer enabled."""

    capture_per_side_logs = True

    extra_prefill_env = {
        "SGLANG_DISAGG_STAGING_BUFFER": "1",
        "UCX_TLS": "ze_copy,ze_ipc,ib,tcp",  # Enable ZE transport
    }
    extra_decode_env = {
        "SGLANG_DISAGG_STAGING_BUFFER": "1",
        "SGLANG_DISAGG_STAGING_POOL_SIZE_MB": "512",
        "UCX_TLS": "ze_copy,ze_ipc,ib,tcp",  # Enable ZE transport
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.rdma_devices = []  # NIXL will use ZE transport on XPU
        cls.extra_prefill_args = ["--device", "xpu"]
        # host_pool retraction backup calls cudaHostRegister, which is CUDA-only.
        cls.extra_decode_args = [
            "--device",
            "xpu",
            "--disaggregation-decode-retraction-backup",
            "cpu_tensor",
        ]
        cls.launch_all()

    def test_staging_buffer_allocated_prefill(self):
        """Verify staging buffers are allocated on prefill side."""
        prefill_logs = self._prefill_stderr_buf.getvalue()

        self.assertIn(
            "StagingBuffer allocated",
            prefill_logs,
            "Staging buffer should be allocated on prefill side",
        )
        self.assertIn(
            "method=default allocator",
            prefill_logs,
            "Staging buffer should use default allocator on XPU",
        )
        self.assertIn(
            "ptr_type=XPU-kernel-space",
            prefill_logs,
            "XPU staging buffer should have kernel-space pointer",
        )

    def test_staging_buffer_allocated_decode(self):
        """Verify staging ring buffer is allocated on decode side."""
        decode_logs = self._decode_stderr_buf.getvalue()

        self.assertIn(
            "StagingAllocator (ring+overcommit)",
            decode_logs,
            "Decode side should have ring buffer allocator",
        )
        self.assertIn(
            "512.0 MB",
            decode_logs,
            "Staging pool size should be 512MB as configured",
        )
        self.assertIn(
            "ptr_type=XPU-kernel-space",
            decode_logs,
            "XPU staging allocator should have kernel-space pointer",
        )

    def test_nixl_registered_staging_memory(self):
        """Verify staging memory is registered with NIXL."""
        prefill_logs = self._prefill_stderr_buf.getvalue()
        decode_logs = self._decode_stderr_buf.getvalue()

        self.assertIn(
            "Registered staging memory with NIXL:",
            prefill_logs,
            "Prefill should register staging memory with NIXL",
        )
        self.assertIn(
            "Registered staging memory with NIXL:",
            decode_logs,
            "Decode should register staging memory with NIXL",
        )

    def test_completion_works_with_staging(self):
        """Basic completion should succeed with staging buffer enabled."""
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
        """Deterministic output should be correct with staging buffer."""
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

    def test_staging_buffer_pointer_format(self):
        """Verify XPU staging buffer uses kernel-space pointers (0xffff...)."""
        prefill_logs = self._prefill_stderr_buf.getvalue()
        decode_logs = self._decode_stderr_buf.getvalue()

        import re

        ptr_pattern = r"ptr=0x([0-9a-f]+)"
        prefill_ptrs = re.findall(ptr_pattern, prefill_logs)
        decode_ptrs = re.findall(ptr_pattern, decode_logs)

        self.assertGreater(
            len(prefill_ptrs), 0, "Should find staging buffer pointers in prefill logs"
        )
        self.assertGreater(
            len(decode_ptrs), 0, "Should find staging buffer pointers in decode logs"
        )

        # Bit 63 set shows up as a leading "ffff"; NIXL must treat it as uint64.
        kernel_space_ptrs = [
            p for p in prefill_ptrs + decode_ptrs if p.startswith("ffff")
        ]
        self.assertGreater(
            len(kernel_space_ptrs),
            0,
            f"XPU staging buffers should use kernel-space pointers (0xffff...), "
            f"found: {prefill_ptrs + decode_ptrs}",
        )


if __name__ == "__main__":
    unittest.main()
