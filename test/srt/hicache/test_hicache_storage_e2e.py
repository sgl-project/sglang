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

import requests

from sglang.bench_serving import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestHiCacheStorageE2E(CustomTestCase):
    """Comprehensive E2E tests for HiCache Storage functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and launch server once for all tests"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

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
    def _launch_server_with_hicache(cls, storage_backend="file"):
        """Launch server with HiCache enabled"""
        server_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.8",
            "--hicache-ratio",
            "1.2",
            "--page-size",
            "64",
            "--hicache-storage-backend",
            storage_backend,
            "--enable-cache-report",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--log-level",
            "debug",
        ]

        if storage_backend == "file":
            env_vars = {
                **os.environ,
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            }
        else:
            env_vars = os.environ

        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
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

    # === Core Functionality Tests ===

    def test_01_cache_storage_and_retrieval(self):
        """Test storage and retrieval of large context through remote cache"""
        print("\n=== Testing Large Context Cache Storage & Retrieval ===")

        # Generate substantial context that will be cached
        base_prompt = self.gen_prompt(768)

        # First request - populate cache
        print("Step 1: Populating cache with large context...")
        response1 = self.send_request(base_prompt, max_tokens=150)
        self.assertIsNotNone(response1)

        # Flush device cache to force remote storage access
        print("Step 2: Flushing device cache...")
        self.assertTrue(self.flush_cache(), "Cache flush should succeed")

        # Second request with extended prompt - should hit remote cache
        print("Step 3: Testing cache hit from remote storage...")
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

    def test_02_share_prefix_caching_with_variations(self):
        """Test prefix caching behavior with substantial prompt variations"""
        print("\n=== Testing Prefix Caching with Variations ===")

        # Create a substantial base context
        base_context = self.gen_prompt(512)

        # Establish the base context in cache
        _ = self.send_request(base_context, max_tokens=80)
        self.flush_cache()

        # Variations that should benefit from prefix caching
        variations = [
            "Focus on microservices architecture and containerization.",
            "Emphasize scalability patterns and load balancing strategies.",
            "Discuss data consistency models and distributed consensus algorithms.",
            "Analyze fault tolerance mechanisms and disaster recovery approaches.",
        ]

        for i, variation in enumerate(variations):
            full_prompt = (
                base_context + "\n\n" + variation + "\n\n" + self.gen_prompt(16)
            )
            response = self.send_request(full_prompt, max_tokens=100)
            cached_tokens = self.get_cached_tokens(response)
            print(f"Variation {i+1}: cached_tokens={cached_tokens}")
            self.assertEqual(
                cached_tokens, 512, "Expected prefix cache hit for variation"
            )
            self.assertIsNotNone(response)

    def test_03_multi_turn_conversation_caching(self):
        """Test caching behavior in multi-turn conversations"""
        print("\n=== Testing Multi-Turn Conversation Caching ===")

        # Simulate a conversation with shared context
        conversation_context = self.gen_prompt(512)

        # Turn 1: Initial question
        turn1_prompt = f"{conversation_context}\n\nUser: Explain the basic concepts."
        response1 = self.send_request(turn1_prompt, max_tokens=160)
        self.assertIsNotNone(response1)
        self.flush_cache()

        # Turn 2: Follow-up question (should benefit from cached context)
        turn2_prompt = f"{conversation_context}\n\nUser: Explain the basic concepts.{response1['text']}\n\nUser: Can you provide more details?"
        response2 = self.send_request(turn2_prompt, max_tokens=80)
        cached_tokens2 = self.get_cached_tokens(response2)
        print(f"Turn 2 cached_tokens: {cached_tokens2}")
        self.assertEqual(
            cached_tokens2, 512, "Expected cache hit for conversation continuation"
        )
        self.flush_cache()

        # Turn 3: Another follow-up (should benefit even more)
        turn3_prompt = f"{turn2_prompt}\n\nUser: {response2['text']}\n\nUser: What are the practical applications?"
        response3 = self.send_request(turn3_prompt, max_tokens=80)
        cached_tokens3 = self.get_cached_tokens(response3)
        print(f"Turn 3 cached_tokens: {cached_tokens3}")
        self.assertGreater(
            cached_tokens3, 600, "Expected cache hit for conversation continuation"
        )

    def test_04_data_persistence_across_node_restart(self):
        """Test data persistence and sharing capability across node restarts"""
        print("\n=== Testing Data Persistence Across Node Restart ===")

        # Phase 1: Start first node and populate cache with substantial data
        print("Phase 1: Starting first node and populating cache...")

        # Generate multiple substantial contexts to ensure meaningful storage
        contexts = []
        for i in range(3):
            context = self.gen_prompt(512)
            contexts.append(context)

        # Send requests to populate cache
        responses = []
        for i, context in enumerate(contexts):
            response = self.send_request(context, max_tokens=80)
            responses.append(response)
            self.assertIsNotNone(response)
            print(
                f"Populated context {i+1} with response length: {len(response['text'])}"
            )

        # Verify cache is working on first node
        print("Verifying cache functionality on first node...")
        for i, context in enumerate(contexts):
            extended_context = context + "\n\nAdditional query for verification."
            response = self.send_request(extended_context, max_tokens=60)
            cached_tokens = self.get_cached_tokens(response)
            print(f"First node - Context {i+1} cached_tokens: {cached_tokens}")
            self.assertEqual(cached_tokens, 512, "Expected cache hit on first node")

        # Phase 2: Kill the first node
        print("Phase 2: Killing first node...")
        if hasattr(self, "process") and self.process:
            kill_process_tree(self.process.pid)
            self.process = None
            print("First node killed successfully")

        time.sleep(2)

        # Phase 3: Start a new node with the same storage backend
        print("Phase 3: Starting new node with same storage configuration...")
        self.process = self._launch_server_with_hicache()
        self._wait_for_server_ready()
        print("New node started successfully")

        # Phase 4: Verify data persistence by testing cache hits
        print("Phase 4: Verifying data persistence on new node...")

        # Test cache hits for the same contexts
        for i, context in enumerate(contexts):
            extended_context = context + "\n\nPersistence verification query."
            response = self.send_request(extended_context, max_tokens=60)
            cached_tokens = self.get_cached_tokens(response)
            print(f"New node - Context {i+1} cached_tokens: {cached_tokens}")

            # Should have significant cache hit from persistent storage
            self.assertEqual(
                cached_tokens, 512, f"Expected cache hit on new node for context {i+1}"
            )

        kill_process_tree(self.process.pid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
