"""
Benchmark tests for HiCache Storage with 3FS backend.
Usage:
    python3 -m pytest test/srt/hicache/test_hicache_storage_3fs_backend.py -v
"""

import json
import os
import time
import unittest
from types import SimpleNamespace

from test_hicache_storage_file_backend import HiCacheStorageBaseMixin

from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import CustomTestCase


class HiCacheStorage3FSBackendBaseMixin(HiCacheStorageBaseMixin):
    """Base mixin class with common setup and utilities"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        # Create a temporary JSON config file for HF3FS
        hf3fs_config = {
            "file_path_prefix": os.path.join(cls.temp_dir, "hicache"),
            "file_size": 1024 * 1024 * 1024 * 2,
            "numjobs": 2,
            "entries": 8,
            "use_mock_hf3fs_client": True,
        }

        # Write config to temporary file
        config_file = os.path.join(cls.temp_dir, "hf3fs_config.json")
        with open(config_file, "w") as f:
            json.dump(hf3fs_config, f, indent=2)

        server_args = {
            "--tp-size": 1,
            "--hicache-ratio": 1.2,
            "--hicache-storage-backend": "hf3fs",
            "--hicache-storage-backend-extra-config": json.dumps(hf3fs_config),
        }

        # Set the environment variable to point to our config file
        env_vars = {
            "SGLANG_HICACHE_HF3FS_CONFIG_PATH": config_file,
        }

        return server_args, env_vars


class TestHf3fsBackendLayerFirstLayout(
    HiCacheStorage3FSBackendBaseMixin, CustomTestCase
):
    """Layer first layout tests for HiCache-Hf3fs backend"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-mem-layout"] = "layer_first"
        server_args["--hicache-io-backend"] = "direct"
        return server_args, env_vars


class TestHf3fsBackendPageFirstLayout(
    HiCacheStorage3FSBackendBaseMixin, CustomTestCase
):
    """Page first layout tests for HiCache-Hf3fs backend"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-mem-layout"] = "page_first"
        return server_args, env_vars


class TestHf3fsBackendAccuracy(HiCacheStorage3FSBackendBaseMixin, CustomTestCase):
    """Accuracy tests for HiCache-Hf3fs backend"""

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
