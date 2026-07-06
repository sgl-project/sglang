"""
E2E test for HiCache mooncake storage with EAGLE3 speculative decoding.

Verifies that after a server restart, both target and draft KV cache for a
long deterministic prompt are reloaded from the mooncake_master process and
the spec accept length does not regress.

Mooncake_master + http metadata server lifecycle is reused from
``test_hicache_storage_mooncake_backend.HiCacheStorageMooncakeBackendBaseMixin``;
this file only overrides the SGLang-side setup for EAGLE3 + spec loadback.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_spec_mooncake_storage.py -v
"""

import os
import signal
import socket
import subprocess
import time
import unittest

from test_hicache_storage_mooncake_backend import (
    HiCacheStorageMooncakeBackendBaseMixin,
)

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.hicache_spec_storage_common import HiCacheSpecStorageMixin
from sglang.test.test_utils import CustomTestCase, find_available_port

register_cuda_ci(est_time=240, stage="extra-a", runner_config="2-gpu-large")


@unittest.skipIf(
    is_hip(), "HiCache + EAGLE3 mooncake-storage loadback e2e is CUDA-only."
)
class TestHiCacheSpecMooncakeStorage(
    HiCacheSpecStorageMixin, HiCacheStorageMooncakeBackendBaseMixin, CustomTestCase
):
    """EAGLE3 + mooncake L3 loadback. After a server restart the long
    deterministic prompt's target+draft KV must be reloaded from
    mooncake_master and the spec accept length must not regress."""

    storage_backend = "mooncake"
    expected_storage_backend = "MooncakeStore"
    # Mooncake exposes no external page count; wait this long after the
    # first prompt for HiCacheController's backup queue to drain into the
    # mooncake_master before restart.
    mooncake_backup_drain_seconds = 15
    mooncake_store_port_base = 50052
    mooncake_store_http_port_base = 8081

    # The inherited test from HiCacheStorageBaseMixin assumes a basic-storage
    # setup; it is already covered by test_hicache_storage_mooncake_backend.py.
    @unittest.skip("Covered by test_hicache_storage_mooncake_backend.py")
    def test_basic_backup_and_prefetch(self):  # noqa: D401
        pass

    @classmethod
    def setUpClass(cls):
        # Bypass the parent chain's basic-storage server bootstrap; only
        # reuse the inherited mooncake_master / metadata-server lifecycle.
        cls.mooncake_master_port = find_available_port(cls.mooncake_master_port_base)
        cls.mooncake_metadata_port = find_available_port(
            cls.mooncake_metadata_port_base
        )
        cls.mooncake_store_port = find_available_port(cls.mooncake_store_port_base)
        cls.mooncake_store_http_port = find_available_port(
            cls.mooncake_store_http_port_base
        )
        cls._start_mooncake_services()
        try:
            cls._start_mooncake_store_service()
            cls._launch_spec_server()
        except Exception:
            cls._stop_mooncake_store_service()
            cls._stop_mooncake_services()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            cls._stop_spec_server()
        finally:
            cls._stop_mooncake_store_service()
            cls._stop_mooncake_services()

    # ---- server side ----

    @classmethod
    def _start_mooncake_store_service(cls):
        print(
            f"Starting Mooncake store service on rpc port {cls.mooncake_store_port}, "
            f"http port {cls.mooncake_store_http_port}..."
        )
        cls.store_service_process = subprocess.Popen(
            [
                "mooncake_client",
                "--host=127.0.0.1",
                f"--port={cls.mooncake_store_port}",
                f"--master_server_address=127.0.0.1:{cls.mooncake_master_port}",
                f"--metadata_server=http://127.0.0.1:{cls.mooncake_metadata_port}/metadata",
                "--protocol=tcp",
                "--device_names=",
                "--global_segment_size=4294967296",
                "--enable_http_server=true",
                f"--http_port={cls.mooncake_store_http_port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        cls._wait_for_mooncake_store_service_ready()

    @classmethod
    def _wait_for_mooncake_store_service_ready(cls, timeout: int = 90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process = getattr(cls, "store_service_process", None)
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"Mooncake store service exited with code {process.returncode}"
                )
            try:
                with socket.create_connection(
                    ("127.0.0.1", cls.mooncake_store_port), timeout=2
                ):
                    pass
                print("Mooncake store service is ready")
                return
            except OSError:
                pass
            time.sleep(1)
        raise TimeoutError("Timed out waiting for Mooncake store service readiness.")

    @classmethod
    def _stop_mooncake_store_service(cls):
        process = getattr(cls, "store_service_process", None)
        if process is None:
            return
        print("Stopping Mooncake store service...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=15)
            print("Mooncake store service stopped")
        except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired, OSError) as e:
                print(f"Warning: Could not stop Mooncake store service: {e}")
        finally:
            cls.store_service_process = None

    @classmethod
    def _get_spec_server_env(cls):
        return {
            "MOONCAKE_MASTER": f"127.0.0.1:{cls.mooncake_master_port}",
            "MOONCAKE_PROTOCOL": "tcp",
            "MC_MS_AUTO_DISC": "0",
            "MOONCAKE_DEVICE": "",
            "MOONCAKE_TE_META_DATA_SERVER": (
                f"http://127.0.0.1:{cls.mooncake_metadata_port}/metadata"
            ),
            # Keep storage capacity in the external mooncake_client so cached
            # pages survive SGLang server restart.
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "0",
        }

    # ---- prompt / IO helpers ----

    def _wait_for_storage_before_restart(self):
        print(
            f"[mooncake] draining backup queue "
            f"({self.mooncake_backup_drain_seconds}s)..."
        )
        time.sleep(self.mooncake_backup_drain_seconds)

    # ---- the test ----

    def test_mooncake_storage_loadback_keeps_spec_accept_length(self):
        self._run_storage_loadback_keeps_spec_accept_length()


if __name__ == "__main__":
    unittest.main(verbosity=2)
