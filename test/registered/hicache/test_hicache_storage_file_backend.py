"""
E2E tests for HiCache Storage functionality.
Usage:
    python3 -m pytest test/registered/hicache/test_hicache_storage_file_backend.py -v
"""

import json
import os
import random
import tempfile
import time
import unittest
from types import SimpleNamespace
from typing import Dict
from urllib.parse import urlparse

import requests

from sglang.bench_serving import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=200, suite="stage-b-test-large-2-gpu")
register_amd_ci(est_time=526, suite="stage-b-test-large-2-gpu-amd")


class HiCacheStorageBaseMixin:
    """Base mixin class with common setup and utilities"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and launch server once for all tests"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = cls._get_model_name()
        cls.base_url = DEFAULT_URL_FOR_TEST

        parsed_url = urlparse(cls.base_url)
        cls.base_host = parsed_url.hostname
        cls.base_port = str(parsed_url.port)

        # Prepare tokenizer for prompt generation
        cls.tokenizer = get_tokenizer(cls.model)

        # Launch server with HiCache enabled and cache report
        cls.process = cls._launch_server_with_hicache()
        cls._wait_for_server_ready()

        print(f"Test server launched successfully at {cls.base_url}")
        print(f"Cache directory: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        kill_process_tree(cls.process.pid)

        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _get_model_name(cls):
        """Get model name for the test configuration - override in subclasses"""
        return DEFAULT_MODEL_NAME_FOR_TEST

    @classmethod
    def _get_base_server_args(cls):
        """Get base server arguments - can be extended in subclasses"""
        extra_config = {
            "hicache_storage_pass_prefix_keys": True,
        }
        return {
            "--enable-hierarchical-cache": True,
            "--mem-fraction-static": 0.6,
            "--hicache-ratio": 1.2,
            "--page-size": 64,
            "--enable-cache-report": True,
            "--hicache-storage-prefetch-policy": "wait_complete",
            "--hicache-storage-backend": "file",
            "--hicache-storage-backend-extra-config": json.dumps(extra_config),
        }

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        return {}, {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}

    @classmethod
    def _launch_server_with_hicache(cls):
        """Launch server with HiCache enabled"""

        additional_server_args, env_vars = cls._get_additional_server_args_and_env()
        env_vars["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "1"
        server_args = cls._get_base_server_args()
        if additional_server_args:
            server_args.update(additional_server_args)

        final_server_args = []
        for k, v in server_args.items():
            if isinstance(v, bool):
                final_server_args.append(str(k))
            else:
                final_server_args.append(str(k))
                final_server_args.append(str(v))

        print(f"final_server_args: {final_server_args}")

        env_vars = {
            **os.environ,
            **env_vars,
        }

        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=final_server_args,
            env=env_vars,
        )

    @classmethod
    def _wait_for_server_ready(cls, timeout: int = 60) -> bool:
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{cls.base_url}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError("Server failed to start within timeout")

    def send_request(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0
    ) -> Dict:
        """Send a generate request and return response"""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def get_cached_tokens(self, response_json: Dict) -> int:
        """Extract cached tokens count from /generate response"""
        meta = response_json.get("meta_info", {})
        return int(meta.get("cached_tokens", 0))

    def flush_cache(self) -> bool:
        """Flush device cache to force remote storage access"""
        try:
            response = requests.post(f"{self.base_url}/flush_cache", timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def gen_prompt(self, token_num: int) -> str:
        """Generate a random prompt of specified token length using tokenizer vocabulary."""
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def trigger_offloading_and_flush(self):
        """Helper method to trigger offloading and flush cache"""
        # Trigger offloading
        self.send_request(self.gen_prompt(1), max_tokens=150)

        # Flush device cache to force remote storage access
        time.sleep(2)
        self.assertTrue(self.flush_cache(), "Cache flush should succeed")

    def test_basic_backup_and_prefetch(self):
        """Test storage and retrieval of large context through remote cache"""
        print("\n=== Testing Large Context Cache Storage & Retrieval ===")

        # Generate substantial context that will be cached
        base_prompt = self.gen_prompt(768)

        # First request - populate cache
        print("Step 1: Populating cache with large context...")
        response1 = self.send_request(base_prompt, max_tokens=150)
        self.assertIsNotNone(response1)

        # Flush device cache to force remote storage access
        self.trigger_offloading_and_flush()

        # Second request with extended prompt - should hit remote cache
        print("Step 2: Testing cache hit from remote storage...")

        start_time = time.time()
        response2 = self.send_request(base_prompt, max_tokens=150)
        retrieval_time = time.time() - start_time

        cached_tokens = self.get_cached_tokens(response2)
        print(
            f"Remote cache retrieval time: {retrieval_time:.3f}s, cached_tokens={cached_tokens}"
        )

        # Assert cached tokens indicate a remote hit
        self.assertGreater(
            cached_tokens, 700, "Expected significant cached tokens for remote hit"
        )


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestHiCacheStoragePageFirstLayout(HiCacheStorageBaseMixin, CustomTestCase):
    """Page first layout tests for HiCache Storage functionality"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args = {"--hicache-mem-layout": "page_first"}
        return server_args, {}


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestHiCacheStorageMLA(HiCacheStorageBaseMixin, CustomTestCase):
    """MLA Model tests for HiCache Storage functionality"""

    @classmethod
    def _get_model_name(cls):
        """Use MLA model for testing"""
        return DEFAULT_MLA_MODEL_NAME_FOR_TEST

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args = {"--tp-size": 2}
        return server_args, {}


class TestHiCacheStoragePageFirstDirectIO(HiCacheStorageBaseMixin, CustomTestCase):
    """Page first direct tests for HiCache Storage functionality"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args = {
            "--hicache-mem-layout": "page_first_direct",
            "--hicache-io-backend": "direct",
            "--tp-size": 2,
        }
        return server_args, {}


class TestHiCacheStorageAccuracy(HiCacheStorageBaseMixin, CustomTestCase):
    """Accuracy tests for HiCache Storage functionality"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args = {
            "--tp-size": 2,
            "--hicache-ratio": 1.5,
        }

        return server_args, {}

    def test_eval_accuracy(self):
        """Test eval accuracy with cache persistence across cache flushes"""
        run_eval_accuracy_test(self)


def run_eval_accuracy_test(test_instance, accuracy_threshold: float = 0.03):
    """Generic eval accuracy test with configurable accuracy threshold

    Args:
        test_instance: The test class instance that provides base_host, base_port, flush_cache, and assert methods
    """
    print("\n=== Testing Eval Accuracy with Cache Persistence ===")

    # First evaluation - populate cache
    print("Phase 1: Running initial GSM8K evaluation to populate cache...")
    args_initial = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=50,
        max_new_tokens=512,
        parallel=10,
        host=f"http://{test_instance.base_host}",
        port=int(test_instance.base_port),
    )
    metrics_initial = run_eval_few_shot_gsm8k(args_initial)

    # Flush cache to force remote storage access
    print("Phase 2: Flushing device cache...")
    test_instance.assertTrue(test_instance.flush_cache(), "Cache flush should succeed")
    time.sleep(2)

    # Second evaluation - should use remote cache
    print("Phase 3: Running second GSM8K evaluation using remote cache...")
    metrics_cached = run_eval_few_shot_gsm8k(args_initial)

    # Verify accuracy consistency
    accuracy_diff = abs(metrics_initial["accuracy"] - metrics_cached["accuracy"])
    print(f"Accuracy difference: {accuracy_diff:.4f}")

    # Assertions
    test_instance.assertGreater(
        metrics_initial["accuracy"], 0.6, "Initial accuracy should be reasonable"
    )
    test_instance.assertGreater(
        metrics_cached["accuracy"], 0.6, "Cached accuracy should be reasonable"
    )
    test_instance.assertLess(
        accuracy_diff,
        accuracy_threshold,
        "Accuracy should be consistent between cache states",
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
