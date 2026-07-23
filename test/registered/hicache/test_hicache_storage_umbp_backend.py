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
import shutil
import signal
import socket
import subprocess
import tempfile
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


def _find_available_port_range(start: int, count: int = 1) -> int:
    """Return the first base port whose whole contiguous range is free."""
    for base in range(start, 65536 - count):
        sockets = []
        try:
            for port in range(base, base + count):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("", port))
                sockets.append(sock)
            return base
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError(f"Could not find {count} consecutive ports from {start}.")


@unittest.skipUnless(is_hip(), "UMBP HiCache requires ROCm.")
@unittest.skipUnless(
    os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "unified_kv_triton")
    == "unified_kv_triton",
    "UMBP HiCache E2E only runs in the unified_kv_triton DSV4 nightly leg.",
)
class TestHiCacheStorageUMBPBackend(CustomTestCase):
    """DeepSeek-V4 hybrid HostPoolGroup round trip through UMBP L3."""

    input_ids = list(range(4000, 5024))

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_FLASH_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.temp_dir = tempfile.mkdtemp(prefix="hicache_umbp_e2e_")
        cls.process = None
        cls.master_process = None
        cls.master_log_file = None

        cls.master_port = _find_available_port_range(59151)
        cls.master_http_port = _find_available_port_range(cls.master_port + 1)
        cls.io_engine_port = _find_available_port_range(19600, TP_SIZE)
        cls.peer_service_port = _find_available_port_range(19700, TP_SIZE)

        try:
            cls._start_umbp_master()
            cls._launch_server()
        except Exception:
            cls._stop_server()
            cls._stop_umbp_master()
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
            raise

    @classmethod
    def tearDownClass(cls):
        cls._stop_server()
        cls._stop_umbp_master()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _start_umbp_master(cls):
        # Importing mori.umbp discovers the master bundled in the wheel and sets
        # UMBP_MASTER_BIN. Keep PATH lookup for development images.
        try:
            import mori.umbp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("mori.umbp is required for the UMBP E2E test.") from exc

        master_bin = os.environ.get("UMBP_MASTER_BIN") or shutil.which("umbp_master")
        if not master_bin:
            raise RuntimeError(
                "UMBP master binary was not found in UMBP_MASTER_BIN or PATH."
            )

        master_log_path = os.path.join(cls.temp_dir, "umbp_master.log")
        cls.master_log_file = open(master_log_path, "w")
        master_env = os.environ.copy()
        master_lib_dir = os.path.dirname(os.path.realpath(master_bin))
        existing_ld_path = master_env.get("LD_LIBRARY_PATH")
        master_env["LD_LIBRARY_PATH"] = (
            f"{master_lib_dir}:{existing_ld_path}"
            if existing_ld_path
            else master_lib_dir
        )
        cls.master_process = subprocess.Popen(
            [
                master_bin,
                f"0.0.0.0:{cls.master_port}",
                str(cls.master_http_port),
            ],
            stdout=cls.master_log_file,
            stderr=subprocess.STDOUT,
            env=master_env,
            preexec_fn=os.setsid,
        )

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if cls.master_process.poll() is not None:
                raise RuntimeError(
                    "umbp_master exited during startup; "
                    f"see {master_log_path} (rc={cls.master_process.returncode})."
                )
            try:
                with socket.create_connection(
                    ("127.0.0.1", cls.master_port), timeout=1
                ):
                    return
            except OSError:
                time.sleep(0.5)
        raise TimeoutError(
            f"Timed out waiting for umbp_master on port {cls.master_port}; "
            f"see {master_log_path}."
        )

    @classmethod
    def _stop_umbp_master(cls):
        process = getattr(cls, "master_process", None)
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
                    pass
        cls.master_process = None

        log_file = getattr(cls, "master_log_file", None)
        if log_file is not None:
            log_file.close()
        cls.master_log_file = None

    @classmethod
    def _launch_server(cls):
        storage_config = {
            "dram_capacity_bytes": 8 * 1024 * 1024 * 1024,
            "ssd_enabled": False,
            "master_address": f"127.0.0.1:{cls.master_port}",
            "io_engine_port": cls.io_engine_port,
            "peer_service_port": cls.peer_service_port,
            "cache_remote_fetches": False,
            # CI runners do not reserve a ~TB hugetlb pool. Registering each
            # complete DSV4 host side-pool as one 4-KiB-backed MR can exhaust
            # the NIC's MTT resources, so use UMBP's bounded staging path here.
            # The same batch_*_v2 keys, existence checks, and payloads are used.
            "disable_zero_copy_register": True,
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
            # Give UMBP clients a chance to unregister their host buffers before
            # the process tree is force-killed.
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
