"""
Benchmark tests for HiCache Storage with Mooncake backend.
Usage:
    python3.10 -m pytest test/srt/hicache/test_hicache_storage_mooncake_backend.py -v
"""

import json
import os
import subprocess
import time
import unittest
from types import SimpleNamespace

import requests
from test_hicache_storage_file_backend import HiCacheStorageBaseMixin

from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_disaggregation_utils import get_rdma_devices_args
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    find_available_port,
    is_in_ci,
)


class HiCacheStorageMooncakeBackendBaseMixin(HiCacheStorageBaseMixin):
    """Base mixin class with common setup and utilities"""

    # Default port ranges for Mooncake services - can be overridden in subclasses
    mooncake_master_port_base = 50051
    mooncake_metadata_port_base = 8080

    @classmethod
    def setUpClass(cls):
        """Set up test environment and launch Mooncake services before server setup"""
        # Find available ports for Mooncake services to avoid conflicts
        cls.mooncake_master_port = find_available_port(
            HiCacheStorageMooncakeBackendBaseMixin.mooncake_master_port_base
        )
        cls.mooncake_metadata_port = find_available_port(
            HiCacheStorageMooncakeBackendBaseMixin.mooncake_metadata_port_base
        )

        # Start Mooncake services first
        cls._start_mooncake_services()

        # Call parent setup
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        """Clean up Mooncake services after server teardown"""
        # Call parent teardown first
        super().tearDownClass()

        # Stop Mooncake services
        cls._stop_mooncake_services()

    @classmethod
    def _start_mooncake_services(cls):
        """Start Mooncake metadata and master services with configurable ports and readiness detection"""
        print("Starting Mooncake services...")
        print(
            f"Using master port: {cls.mooncake_master_port}, metadata port: {cls.mooncake_metadata_port}"
        )

        # Start metadata service with configurable port
        try:
            # Start metadata server with port configuration
            cls.metadata_service_process = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "mooncake.http_metadata_server",
                    "--port",
                    str(cls.mooncake_metadata_port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,  # Create new process group
            )
            print(
                f"Mooncake metadata service started on port {cls.mooncake_metadata_port}"
            )
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"Warning: Could not start Mooncake metadata service: {e}")
            cls.metadata_service_process = None

        # Start master service with configurable port
        try:
            # Start master server with port configuration
            cls.master_service_process = subprocess.Popen(
                ["mooncake_master", "--port", str(cls.mooncake_master_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,  # Create new process group
            )
            print(f"Mooncake master service started on port {cls.mooncake_master_port}")
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"Warning: Could not start Mooncake master service: {e}")
            cls.master_service_process = None

        # Wait for services to be ready instead of fixed sleep
        cls._wait_for_mooncake_services_ready()

    @classmethod
    def _wait_for_mooncake_services_ready(cls, timeout: int = 30) -> bool:
        """Wait for Mooncake services to be ready by checking their endpoints"""
        print("Waiting for Mooncake services to be ready...")

        start_time = time.time()
        services_ready = False

        while time.time() - start_time < timeout:
            try:
                # Check metadata service
                metadata_ready = False
                if (
                    cls.metadata_service_process
                    and cls.metadata_service_process.poll() is None
                ):
                    try:
                        # Try to connect to the metadata service
                        metadata_url = (
                            f"http://127.0.0.1:{cls.mooncake_metadata_port}/metadata"
                        )
                        response = requests.get(metadata_url, timeout=2)
                        if response.status_code == 200:
                            metadata_ready = True
                            print("Mooncake metadata service is ready")
                    except (requests.RequestException, ConnectionError):
                        # Service might not be fully started yet
                        pass

                # Check master service (if it has a health endpoint)
                master_ready = False
                if (
                    cls.master_service_process
                    and cls.master_service_process.poll() is None
                ):
                    # For now, we'll assume master service is ready if process is running
                    # and it's been a few seconds since startup
                    if (
                        time.time() - start_time > 5
                    ):  # Give master service time to initialize
                        master_ready = True
                        print("Mooncake master service is ready")

                # Both services should be ready
                if metadata_ready and master_ready:
                    services_ready = True
                    print("All Mooncake services are ready")
                    break

            except Exception as e:
                print(f"Error checking service readiness: {e}")

            time.sleep(2)

        if not services_ready:
            print(
                "Warning: Mooncake services may not be fully ready, continuing anyway..."
            )

        return services_ready

    @classmethod
    def _stop_mooncake_services(cls):
        """Stop Mooncake services"""
        print("Stopping Mooncake services...")

        # Stop metadata service
        if hasattr(cls, "metadata_service_process") and cls.metadata_service_process:
            try:
                os.killpg(os.getpgid(cls.metadata_service_process.pid), 9)
                cls.metadata_service_process.wait(timeout=5)
                print("Mooncake metadata service stopped")
            except (ProcessLookupError, subprocess.TimeoutExpired, OSError) as e:
                print(f"Warning: Could not stop Mooncake metadata service: {e}")

        # Stop master service
        if hasattr(cls, "master_service_process") and cls.master_service_process:
            try:
                os.killpg(os.getpgid(cls.master_service_process.pid), 9)
                cls.master_service_process.wait(timeout=5)
                print("Mooncake master service stopped")
            except (ProcessLookupError, subprocess.TimeoutExpired, OSError) as e:
                print(f"Warning: Could not stop Mooncake master service: {e}")

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""

        server_args = {
            "--tp-size": 2,
            "--hicache-ratio": 2,
            "--hicache-storage-backend": "mooncake",
        }

        # Set the environment variables for Mooncake using dynamic ports
        env_vars = {
            "MOONCAKE_MASTER": f"127.0.0.1:{cls.mooncake_master_port}",
            "MOONCAKE_PROTOCOL": "tcp",
            "MC_MS_AUTO_DISC": "0",
            "MOONCAKE_DEVICE": "",
            "MOONCAKE_TE_META_DATA_SERVER": f"http://127.0.0.1:{cls.mooncake_metadata_port}/metadata",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "4294967296",  # 4 GiB
        }

        return server_args, env_vars


'''
# Same as #10131, layer first layout test TODO(mateng): will make it work
class TestMooncakeBackendLayerFirstLayout(
    HiCacheStorageMooncakeBackendBaseMixin, CustomTestCase
):
    """Layer first layout tests for HiCache-Mooncake backend"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-mem-layout"] = "layer_first"
        server_args["--hicache-io-backend"] = "direct"
        return server_args, env_vars
'''


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestMooncakeBackendPageFirstLayout(
    HiCacheStorageMooncakeBackendBaseMixin, CustomTestCase
):
    """Page first layout tests for HiCache-Mooncake backend"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-mem-layout"] = "page_first"
        return server_args, env_vars


class TestMooncakeBackendMLAModel(
    HiCacheStorageMooncakeBackendBaseMixin, CustomTestCase
):
    """MLA Model tests for HiCache-Mooncake backend"""

    @classmethod
    def _get_model_name(cls):
        """Use MLA model for testing"""
        return DEFAULT_MLA_MODEL_NAME_FOR_TEST

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-mem-layout"] = "page_first"
        server_args["--tp-size"] = 2
        return server_args, env_vars


class TestMooncakeBackendAccuracy(
    HiCacheStorageMooncakeBackendBaseMixin, CustomTestCase
):
    """Accuracy tests for HiCache-Mooncake backend"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-ratio"] = 1.5
        server_args["--tp-size"] = 2
        server_args["--hicache-mem-layout"] = "page_first_direct"
        server_args["--hicache-io-backend"] = "direct"
        return server_args, env_vars

    def test_eval_accuracy(self):
        """Test eval accuracy with cache persistence across cache flushes"""
        from test_hicache_storage_file_backend import run_eval_accuracy_test

        run_eval_accuracy_test(self)


if __name__ == "__main__":
    unittest.main(verbosity=2)
