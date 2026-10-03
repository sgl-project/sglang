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
import tempfile
import time
import unittest
from pathlib import Path

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
UMBP_SERVER_LAUNCH_TIMEOUT = 60
UMBP_SERVER_READY_MARKER = (
    "[StandaloneServer] data plane: serialized reads (SSD medium)"
)
UMBP_STORAGE_PAGE_SIZE = 2 * 1024 * 1024
UMBP_SSD_CAPACITY = 20 * 1024 * 1024 * 1024
PAGE_SIZE = 256
TP_SIZE = 8


@unittest.skipUnless(is_hip(), "UMBP HiCache requires ROCm.")
@unittest.skipUnless(
    os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "unified_kv_triton")
    == "unified_kv_triton",
    "UMBP HiCache E2E only runs in the unified_kv_triton DSV4 nightly leg.",
)
class TestHiCacheStorageUMBPBackend(CustomTestCase):
    """DeepSeek-V4 hybrid HostPoolGroup round trip through UMBP SSD."""

    input_ids = list(range(4000, 5024))

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_FLASH_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None
        cls.umbp_process = None
        cls.umbp_temp_dir = None
        cls.umbp_log_handle = None

        try:
            cls._launch_umbp_server()
            cls._launch_server()
        except BaseException:
            cls._stop_server()
            raise

    @classmethod
    def tearDownClass(cls):
        cls._stop_server()

    @classmethod
    def _launch_umbp_server(cls):
        # Importing the package resolves the standalone binary built alongside
        # the Python extension and publishes it via UMBP_STANDALONE_BIN.
        import mori.umbp  # noqa: F401

        server_bin = os.environ.get("UMBP_STANDALONE_BIN", "")
        if not server_bin or not os.access(server_bin, os.X_OK):
            raise RuntimeError(
                f"UMBP_STANDALONE_BIN is unset or not executable: {server_bin!r}"
            )

        # Keep the UDS path short: AF_UNIX paths are limited to 108 bytes.
        cls.umbp_temp_dir = tempfile.TemporaryDirectory(
            prefix="sglang-umbp-", dir="/tmp"
        )
        root = Path(cls.umbp_temp_dir.name)
        cls.umbp_ssd_dir = root / "ssd"
        cls.umbp_ssd_dir.mkdir()
        cls.umbp_address = f"unix://{root}/node.grpc.sock"
        cls.umbp_log_path = root / "umbp-server.log"

        server_env = os.environ.copy()
        for name in (
            "UMBP_MASTER_ADDRESS",
            "UMBP_NODE_ADDRESS",
            "UMBP_NODE_ID",
            "UMBP_IO_ENGINE_HOST",
            "UMBP_IO_ENGINE_PORT",
            "UMBP_PEER_SERVICE_PORT",
            "UMBP_BACKEND_POLICY",
            "UMBP_STANDALONE_ADDRESS",
            "UMBP_STANDALONE_AUTO_START",
        ):
            server_env.pop(name, None)
        server_env.update(
            {
                "UMBP_STANDALONE_ADDRESS": cls.umbp_address,
                "UMBP_ROLE": "standalone",
                "UMBP_DISTRIBUTED_MEDIUM": "SSD",
                "UMBP_DISTRIBUTED_DRAM_PAGE_SIZE": str(UMBP_STORAGE_PAGE_SIZE),
                "UMBP_SSD_ENABLED": "1",
                "UMBP_SSD_BACKEND": "file",
                "UMBP_SSD_DIR": str(cls.umbp_ssd_dir),
                "UMBP_SSD_CAPACITY": str(UMBP_SSD_CAPACITY),
                "UMBP_DRAM_USE_HUGEPAGES": "0",
                "UMBP_DISTRIBUTED_SSD_STAGING_USE_HUGEPAGES": "0",
                "MORI_UMBP_LOG_LEVEL": "info",
                "MORI_GLOBAL_LOG_LEVEL": "info",
            }
        )

        cls.umbp_log_handle = cls.umbp_log_path.open("w", encoding="utf-8")
        cls.umbp_process = subprocess.Popen(
            [server_bin, cls.umbp_address],
            env=server_env,
            stdout=cls.umbp_log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        cls._wait_for_umbp_server()

    @classmethod
    def _wait_for_umbp_server(cls):
        deadline = time.monotonic() + UMBP_SERVER_LAUNCH_TIMEOUT
        while time.monotonic() < deadline:
            if cls.umbp_process.poll() is not None:
                raise RuntimeError(
                    "umbp_standalone_server exited before becoming ready "
                    f"(return code {cls.umbp_process.returncode}):\n"
                    f"{cls._umbp_log_tail()}"
                )
            if UMBP_SERVER_READY_MARKER in cls.umbp_log_path.read_text(
                encoding="utf-8", errors="replace"
            ):
                return
            time.sleep(0.1)

        raise RuntimeError(
            "umbp_standalone_server did not become ready within "
            f"{UMBP_SERVER_LAUNCH_TIMEOUT}s:\n{cls._umbp_log_tail()}"
        )

    @classmethod
    def _umbp_log_tail(cls, line_count=50):
        try:
            lines = cls.umbp_log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError as error:
            return f"(could not read UMBP log: {error})"
        return "\n".join(lines[-line_count:])

    @classmethod
    def _launch_server(cls):
        storage_config = {
            "dram_capacity_bytes": 1 * 1024 * 1024 * 1024,
            "ssd_enabled": True,
            "ssd_storage_dir": str(cls.umbp_ssd_dir),
            "ssd_capacity_bytes": UMBP_SSD_CAPACITY,
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
        # All TP ranks connect to the same out-of-process SSD backend. This is
        # masterless and does not require an RDMA-capable CI runner. Empty
        # values mask stale parent settings when popen_launch_server merges
        # os.environ into this mapping again.
        env.update(
            {
                "UMBP_MASTER_ADDRESS": "",
                "UMBP_NODE_ADDRESS": "",
                "UMBP_NODE_ID": "",
                "UMBP_IO_ENGINE_HOST": "",
                "UMBP_BACKEND_POLICY": "",
                "UMBP_DISTRIBUTED_MEDIUM": "",
                "UMBP_ROLE": "",
                "UMBP_STANDALONE_ADDRESS": cls.umbp_address,
                "UMBP_STANDALONE_AUTO_START": "0",
                "UMBP_STANDALONE_STARTUP_TIMEOUT_MS": "60000",
                "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
                "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
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
        try:
            if process is not None and process.poll() is None:
                # Stop SGLang first so every TP client can deregister its shared
                # memory before the standalone server exits.
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    kill_process_tree(process.pid)
        finally:
            cls.process = None
            cls._stop_umbp_server()

    @classmethod
    def _stop_umbp_server(cls):
        process = getattr(cls, "umbp_process", None)
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    kill_process_tree(process.pid, wait_timeout=10)
        finally:
            cls.umbp_process = None
            log_handle = getattr(cls, "umbp_log_handle", None)
            if log_handle is not None:
                log_handle.close()
                cls.umbp_log_handle = None
            temp_dir = getattr(cls, "umbp_temp_dir", None)
            if temp_dir is not None:
                temp_dir.cleanup()
                cls.umbp_temp_dir = None

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
