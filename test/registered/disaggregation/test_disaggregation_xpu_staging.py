"""
Disaggregation integration test for NIXL backend with staging buffer on Intel XPU.

Tests the staging buffer optimization for KV cache transfer in PD disaggregation.
The staging buffer reduces RDMA request count from O(tokens × layers) to O(1)
by gathering scattered KV head slices into contiguous GPU memory before bulk transfer.

This test verifies:
1. Staging buffer allocation on both prefill and decode sides
2. Level Zero (ZE) transport usage for XPU-to-XPU transfer
3. End-to-end text completion works with staging enabled
4. Staging buffers are properly registered with NIXL

Usage:
    python3 -m pytest test/registered/disaggregation/test_disaggregation_xpu_staging.py -v
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
    est_time=350,
    stage="base-a",
    runner_config="1-gpu-small",  # TODO change it to 2 gpu when test is enabled
    disabled="Intel XPU only — not available in standard CUDA CI",
)

_XPU_AVAILABLE = torch.xpu.is_available()


@unittest.skipUnless(
    _XPU_AVAILABLE, "Intel XPU not available (torch.xpu.is_available() returned False)"
)
class TestDisaggregationNixlStaging(PDDisaggregationServerBase):
    """Test NIXL disaggregation backend with staging buffer enabled."""

    capture_per_side_logs = True  # Capture logs to verify staging buffer usage

    # Enable staging buffer via environment variables
    extra_prefill_env = {
        "SGLANG_DISAGG_STAGING_BUFFER": "1",
        "SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB": "256",
        "UCX_TLS": "ze_copy,ze_ipc,tcp",  # Enable ZE transport
    }
    extra_decode_env = {
        "SGLANG_DISAGG_STAGING_BUFFER": "1",
        "SGLANG_DISAGG_STAGING_POOL_SIZE_MB": "512",
        "UCX_TLS": "ze_copy,ze_ipc,tcp",  # Enable ZE transport
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        # Force NIXL backend and XPU device
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.rdma_devices = []  # NIXL will use ZE transport on XPU
        cls.extra_prefill_args = ["--device", "xpu"]
        cls.extra_decode_args = ["--device", "xpu"]
        subprocess.check_call(
            ["pip", "install", "sglang-router"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cls.launch_all()

    def test_staging_buffer_allocated_prefill(self):
        """Verify staging buffers are allocated on prefill side."""
        prefill_logs = self._prefill_stderr_buf.getvalue()

        # Check for staging buffer allocation messages
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

        # Verify buffer size (should see 256MB per worker)
        self.assertIn(
            "256.0 MB",
            prefill_logs,
            "Staging buffer size should be 256MB as configured",
        )

    def test_staging_buffer_allocated_decode(self):
        """Verify staging ring buffer is allocated on decode side."""
        decode_logs = self._decode_stderr_buf.getvalue()

        # Check for staging allocator (ring buffer) on decode
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

        # Check that memory registration happened
        self.assertIn(
            "Registering staging memory with NIXL:",
            prefill_logs,
            "Prefill should register staging memory with NIXL",
        )
        self.assertIn(
            "Registering staging memory with NIXL:",
            decode_logs,
            "Decode should register staging memory with NIXL",
        )

    def test_nixl_ze_transport_detected(self):
        """Verify NIXL is using Level Zero (ZE) transport for XPU."""
        prefill_logs = self._prefill_stderr_buf.getvalue()

        # Check for ZE transport initialization
        # UCX should detect ze_copy and ze_ipc components
        self.assertIn(
            "ze_",  # Will match ze_copy or ze_ipc or ze_base logs
            prefill_logs,
            "NIXL/UCX should detect Level Zero transport components",
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
        # Model should produce "2" in the output
        self.assertIn("2", generated, f"Expected '2' in output, got: {generated!r}")

    def test_concurrent_requests_with_staging(self):
        """Multiple concurrent requests should work with staging buffer."""
        import concurrent.futures

        def send_request(prompt):
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0.8, "max_new_tokens": 50},
                },
            )
            return response.json()

        prompts = [
            "What is machine learning?",
            "Explain quantum computing.",
            "Tell me about artificial intelligence.",
        ]

        # Send 3 requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send_request, p) for p in prompts]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        self.assertEqual(len(results), 3, "Should receive 3 responses")
        for result in results:
            self.assertIn("text", result, "Each response should have 'text' field")
            self.assertGreater(
                len(result["text"]), 0, "Generated text should not be empty"
            )

    def test_staging_buffer_pointer_format(self):
        """Verify XPU staging buffer uses kernel-space pointers (0xffff...)."""
        prefill_logs = self._prefill_stderr_buf.getvalue()
        decode_logs = self._decode_stderr_buf.getvalue()

        # XPU kernel-space pointers have bit 63 set (0xffff...)
        # This is important because it requires uint64 handling in NIXL
        import re

        # Find pointer addresses in logs
        ptr_pattern = r"ptr=0x([0-9a-f]+)"
        prefill_ptrs = re.findall(ptr_pattern, prefill_logs)
        decode_ptrs = re.findall(ptr_pattern, decode_logs)

        self.assertGreater(
            len(prefill_ptrs), 0, "Should find staging buffer pointers in prefill logs"
        )
        self.assertGreater(
            len(decode_ptrs), 0, "Should find staging buffer pointers in decode logs"
        )

        # Check that at least one pointer has bit 63 set (kernel-space)
        # Kernel-space addresses start with 'ffff' in hex
        kernel_space_ptrs = [
            p for p in prefill_ptrs + decode_ptrs if p.startswith("ffff")
        ]
        self.assertGreater(
            len(kernel_space_ptrs),
            0,
            f"XPU staging buffers should use kernel-space pointers (0xffff...), "
            f"found: {prefill_ptrs + decode_ptrs}",
        )


@unittest.skipUnless(_XPU_AVAILABLE, "Intel XPU not available")
class TestDisaggregationNixlStagingPerformance(PDDisaggregationServerBase):
    """Performance-oriented tests for staging buffer."""

    capture_per_side_logs = True

    extra_prefill_env = {
        "SGLANG_DISAGG_STAGING_BUFFER": "1",
        "SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB": "256",
        "UCX_TLS": "ze_copy,ze_ipc,tcp",
    }
    extra_decode_env = {
        "SGLANG_DISAGG_STAGING_BUFFER": "1",
        "SGLANG_DISAGG_STAGING_POOL_SIZE_MB": "512",
        "UCX_TLS": "ze_copy,ze_ipc,tcp",
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
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

    def test_long_sequence_generation(self):
        """Test staging buffer with longer sequences (more benefit expected)."""
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "Write a detailed explanation of how computers work, "
                "covering CPUs, memory, storage, and networking. ",
                "sampling_params": {"temperature": 0.7, "max_new_tokens": 200},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        # Verify completion succeeded
        self.assertIn("text", data)
        self.assertGreater(len(data["text"]), 100, "Long sequence should generate text")

        # Check meta_info for token counts
        if "meta_info" in data:
            meta = data["meta_info"]
            if "completion_tokens" in meta:
                # With staging buffer, longer sequences benefit more
                # (reduces O(tokens × layers) to O(1) RDMA requests)
                self.assertGreater(
                    meta["completion_tokens"],
                    50,
                    "Should generate substantial number of tokens",
                )


if __name__ == "__main__":
    unittest.main()
