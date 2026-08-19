"""E2E test for DeepSeek-V4 HiCache storage with the UMBP backend.

The first request writes the hybrid HostPoolGroup side pools to UMBP. After
flushing the device and host radix caches, the same prompt must be restored
from UMBP and report a storage-tier cache hit.

Usage:
    python3 -m pytest \
        test/registered/hicache/test_hicache_storage_umbp_backend.py -v
"""

import json
import os
import subprocess
import time
import unittest

import requests

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(
    est_time=3600,
    suite="nightly-amd-8-gpu-mi35x-deepseek-v4-flash",
    nightly=True,
)

DEEPSEEK_V4_FLASH_FP8_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_FP8_MODEL_PATH", "sgl-project/DeepSeek-V4-Flash-FP8"
)
SERVER_LAUNCH_TIMEOUT = 3600
PAGE_SIZE = 256
TP_SIZE = 8


@unittest.skipUnless(is_hip(), "UMBP HiCache requires ROCm.")
@unittest.skipUnless(
    os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "unified_kv_triton")
    == "unified_kv_triton",
    "UMBP HiCache E2E only runs in the unified_kv_triton DSV4 nightly leg.",
)
class TestHiCacheStorageUMBPBackend(CustomTestCase):
    """DeepSeek-V4 hybrid HostPoolGroup round trip through local UMBP L3."""

    input_ids = list(range(4000, 5024))

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_FLASH_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None

        try:
            cls._launch_server()
        except Exception:
            cls._stop_server()
            raise

    @classmethod
    def tearDownClass(cls):
        cls._stop_server()

    @classmethod
    def _launch_server(cls):
        storage_config = {
            "dram_capacity_bytes": 1 * 1024 * 1024 * 1024,
            "ssd_enabled": True,
            "ssd_storage_dir": "/tmp/umbp_dsv4_local",
            "ssd_capacity_bytes": 20 * 1024 * 1024 * 1024,
        }
        other_args = [
            "--trust-remote-code",
            "--tp-size",
            str(TP_SIZE),
            "--attention-backend",
            "dsv4",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--page-size",
            str(PAGE_SIZE),
            "--chunked-prefill-size",
            "8192",
            "--mem-fraction-static",
            "0.85",
            "--disable-cuda-graph",
            "--disable-shared-experts-fusion",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "2",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-io-backend",
            "direct",
            "--hicache-mem-layout",
            "page_first",
            "--hicache-storage-backend",
            "mori",
            "--hicache-storage-backend-extra-config",
            json.dumps(storage_config),
            "--enable-cache-report",
            "--enable-metrics",
            "--swa-full-tokens-ratio",
            "0.1",
            "--max-total-tokens",
            "20000",
            "--max-running-requests",
            "4",
            "--watchdog-timeout",
            "1200",
        ]

        env = os.environ.copy()
        # An absent master address keeps every TP rank in standalone local mode,
        # so this E2E does not require an RDMA-capable CI runner.
        env.pop("UMBP_MASTER_ADDRESS", None)
        env.update(
            {
                "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
                "SGLANG_USE_ROCM700A": "0",
                "AITER_BF16_FP8_MOE_BOUND": "0",
                # Correctness does not depend on pre-reserved hugepages, and
                # disabling them makes the E2E portable across MI35x runners.
                "SGLANG_HICACHE_HOST_HUGEPAGE": "0",
                "UMBP_DRAM_USE_HUGEPAGES": "0",
            }
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def _stop_server(cls):
        process = getattr(cls, "process", None)
        if process is None:
            return
        if process.poll() is None:
            # Give UMBP clients a chance to close their local tiers before the
            # process tree is force-killed.
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                kill_process_tree(process.pid)
        cls.process = None

    def _flush_device_and_host_cache(self):
        response = requests.post(
            self.base_url + "/flush_cache",
            params={"timeout": 60},
            timeout=90,
        )
        response.raise_for_status()

    def _generate(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": self.input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
            },
            timeout=1200,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def test_hybrid_host_pool_round_trip_from_umbp(self):
        self._flush_device_and_host_cache()

        first = self._generate()
        self.assertEqual(first["meta_info"]["cached_tokens"], 0)

        # Writes are asynchronous below the request path. This mirrors the
        # Mooncake E2E drain before forcing the next request to use L3.
        time.sleep(15)
        self._flush_device_and_host_cache()

        second = self._generate()
        cached_details = second["meta_info"].get("cached_tokens_details") or {}
        storage_cached_tokens = int(cached_details.get("storage", 0))

        self.assertGreaterEqual(
            storage_cached_tokens,
            PAGE_SIZE,
            "Expected DeepSeek-V4 side-pool KV to load from UMBP storage, "
            f"got {cached_details=}",
        )
        self.assertEqual(cached_details.get("storage_backend"), "UMBPStore")


if __name__ == "__main__":
    unittest.main(verbosity=2)
