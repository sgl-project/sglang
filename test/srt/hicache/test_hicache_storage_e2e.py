"""
E2E tests for HiCache Storage functionality.
Usage:
    python3 -m pytest test/srt/hicache/test_hicache_storage_e2e.py -v
"""

import os
import random
import tempfile
import time
import unittest
from typing import Dict
from urllib.parse import urlparse

import requests

from sglang.bench_serving import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class HiCacheStorageBaseTest(CustomTestCase):
    """Base test class with common setup and utilities"""

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
        return {
            "--enable-hierarchical-cache": True,
            "--mem-fraction-static": 0.6,
            "--hicache-ratio": 1.2,
            "--page-size": 64,
            "--enable-cache-report": True,
            "--hicache-storage-prefetch-policy": "wait_complete",
            "--hicache-storage-backend": "file",
        }

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        return {}, {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}

    @classmethod
    def _launch_server_with_hicache(cls):
        """Launch server with HiCache enabled"""

        additional_server_args, env_vars = cls._get_additional_server_args_and_env()
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
        """Generate a random prompt of specified token length using tokenizer vocabulary.

        This function mimics the implementation from bench_serving.py to create
        realistic prompts for testing cache behavior.
        """
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
        extended_prompt = base_prompt + "\n\n" + self.gen_prompt(64)

        start_time = time.time()
        response2 = self.send_request(extended_prompt, max_tokens=150)
        retrieval_time = time.time() - start_time

        cached_tokens = self.get_cached_tokens(response2)
        print(
            f"Remote cache retrieval time: {retrieval_time:.3f}s, cached_tokens={cached_tokens}"
        )

        # Assert cached tokens indicate a remote hit
        self.assertEqual(
            cached_tokens, 768, "Expected significant cached tokens for remote hit"
        )


class TestHiCacheStorageTP(HiCacheStorageBaseTest):
    """Multi-TP tests for HiCache Storage functionality"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args = {"--tp-size": 2}
        return server_args, {}


class TestHiCacheStorageLayerFirstDirectIO(HiCacheStorageBaseTest):
    """Layer first direct tests for HiCache Storage functionality"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args = {
            "--hicache-mem-layout": "layer_first",
            "--hicache-io-backend": "direct",
        }
        return server_args, {}


class TestHiCacheStoragePageFirstLayout(HiCacheStorageBaseTest):
    """Page first layout tests for HiCache Storage functionality"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args = {"--hicache-mem-layout": "page_first"}
        return server_args, {}


class TestHiCacheStorageMLA(HiCacheStorageBaseTest):
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


# TODO: Add other backends tests（3fs/mooncake）
# class TestHiCacheStorageMooncakeBackend(HiCacheStorageBaseTest):
#     """Mooncake backend tests for HiCache Storage functionality"""

#     @classmethod
#     def _get_additional_server_args_and_env(cls):
#         """Get additional server arguments specific to configuration - override in subclasses"""
#         server_args = ["--hicache-storage-backend", "mooncake"]
#         env = {
#             "MOONCAKE_TE_META_DATA_SERVER": "http://127.0.0.1:8080/metadata",
#             "MOONCAKE_MASTER": "127.0.0.1:50051"
#             xxxxx
#         }
#         return server_args, {}


if __name__ == "__main__":
    unittest.main(verbosity=2)
