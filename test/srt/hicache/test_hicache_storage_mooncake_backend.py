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

from test_hicache_storage_file_backend import HiCacheStorageBaseMixin

from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import CustomTestCase


class HiCacheStorageMooncakeBackendBaseMixin(HiCacheStorageBaseMixin):
    """Base mixin class with common setup and utilities"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and launch Mooncake services before server setup"""
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
        """Start Mooncake metadata and master services"""
        print("Starting Mooncake services...")

        # Start metadata service
        try:
            cls.metadata_service_process = subprocess.Popen(
                ["python3", "-m", "mooncake.http_metadata_server"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,  # Create new process group
            )
            print("Mooncake metadata service started")
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"Warning: Could not start Mooncake metadata service: {e}")
            cls.metadata_service_process = None

        # Start master service
        try:
            cls.master_service_process = subprocess.Popen(
                ["mooncake_master"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,  # Create new process group
            )
            print("Mooncake master service started")
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"Warning: Could not start Mooncake master service: {e}")
            cls.master_service_process = None

        # Give services time to start
        time.sleep(3)

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
            "--tp-size": 1,
            "--hicache-ratio": 1.2,
            "--hicache-storage-backend": "mooncake",
        }

        # Set the environment variables for Mooncake
        env_vars = {
            "MOONCAKE_MASTER": "127.0.0.1:50051",
            "MOONCAKE_PROTOCOL": "rdma",
            "MOONCAKE_DEVICE": "",
            "MOONCAKE_TE_META_DATA_SERVER": "http://127.0.0.1:8080/metadata",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "4294967296",  # 4 GiB
        }

        return server_args, env_vars


# Same as #10131, layer first layout test
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


# Same as #10131, page first layout test
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


# Same as #10131, accuracy test
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
        return server_args, env_vars

    def test_eval_accuracy(self):
        """Test eval accuracy with cache persistence across cache flushes"""
        print("\n=== Testing Eval Accuracy with Cache Persistence ===")

        # First evaluation - populate cache
        print("Phase 1: Running initial GSM8K evaluation to populate cache...")
        args_initial = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=50,
            max_new_tokens=512,
            parallel=10,
            host=f"http://{self.base_host}",
            port=int(self.base_port),
        )
        metrics_initial = run_eval_few_shot_gsm8k(args_initial)

        # Flush cache to force remote storage access
        print("Phase 2: Flushing device cache...")
        self.assertTrue(self.flush_cache(), "Cache flush should succeed")
        time.sleep(2)

        # Second evaluation - should use remote cache
        print("Phase 3: Running second GSM8K evaluation using remote cache...")
        metrics_cached = run_eval_few_shot_gsm8k(args_initial)

        # Verify accuracy consistency
        accuracy_diff = abs(metrics_initial["accuracy"] - metrics_cached["accuracy"])
        print(f"Accuracy difference: {accuracy_diff:.4f}")

        # Assertions
        self.assertGreater(
            metrics_initial["accuracy"], 0.6, "Initial accuracy should be reasonable"
        )
        self.assertGreater(
            metrics_cached["accuracy"], 0.6, "Cached accuracy should be reasonable"
        )
        self.assertLess(
            accuracy_diff, 0.05, "Accuracy should be consistent between cache states"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
