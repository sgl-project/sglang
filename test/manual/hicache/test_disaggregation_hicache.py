import os
import random
import tempfile
import time
import unittest
from typing import Dict

import requests

from sglang.bench_serving import get_tokenizer
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)


class DisaggregationHiCacheBase(PDDisaggregationServerBase):
    """Base class for disaggregation with HiCache tests"""

    @classmethod
    def setUpClass(cls):
        super(DisaggregationHiCacheBase, cls).setUpClass()

        cls.model = DEFAULT_MODEL_NAME_FOR_TEST

        cls.tokenizer = get_tokenizer(cls.model)
        cls.temp_dir = tempfile.mkdtemp()
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        # Prefill with HiCache enabled
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "1",
            "--page-size",
            "64",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "1.2",
            "--hicache-size",
            "0",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--mem-fraction-static",
            "0.8",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls):
        pass

    def gen_prompt(self, token_num: int) -> str:
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def send_request(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0
    ) -> Dict:
        """Send a generate request and return response"""
        response = requests.post(
            f"{self.lb_url}/generate",
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

    def trigger_offloading_and_flush(self):
        """Helper method to trigger offloading and flush cache"""
        # Trigger offloading
        self.send_request(self.gen_prompt(1), max_tokens=150)

        # Flush device cache to force remote storage access
        time.sleep(2)
        requests.post(self.prefill_url + "/flush_cache")


class TestDisaggregationPrefillWithHiCache(DisaggregationHiCacheBase):
    """Test disaggregation with HiCache enabled only on Prefill side"""

    @classmethod
    def start_decode(cls):
        # Decode without HiCache offload
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "1",
            "--page-size",
            "64",
            "--mem-fraction-static",
            "0.8",
            "--base-gpu-id",
            "1",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_prefill_cache_hit(self):
        """Test that prefill cache works with repeated queries"""

        repeated_prompt = self.gen_prompt(800)

        # First request - should miss cache
        self.send_request(repeated_prompt, max_tokens=100)

        # Flush cache
        self.trigger_offloading_and_flush()

        # Second request - should hit cache (faster)
        response2 = self.send_request(repeated_prompt, max_tokens=100)

        # Assert cached tokens cnt
        self.assertGreater(response2["meta_info"]["cached_tokens"], 700)


class TestDisaggregationDecodeWithHiCache(DisaggregationHiCacheBase):
    """Test disaggregation with HiCache enabled on both Prefill and Decode sides"""

    @classmethod
    def start_decode(cls):
        # Decode with HiCache offload enabled
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "1",
            "--page-size",
            "64",
            "--mem-fraction-static",
            "0.8",
            "--base-gpu-id",
            "1",
            "--disaggregation-decode-enable-offload-kvcache",
            "--hicache-ratio",
            "1.2",
            "--hicache-size",
            "0",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_multi_turn_conversation_cache(self):
        """Test multi-turn conversation scenario with cache hit improvement"""

        print("=== Multi-turn Conversation Cache Test ===")

        # Turn 1
        initial_prompt = self.gen_prompt(300)

        response1 = self.send_request(initial_prompt, max_tokens=200, temperature=0.1)
        current_context = initial_prompt + response1["text"]

        # Turns 2-4: Continue generation based on previous context
        previous_cached_tokens = 0

        for turn in range(2, 5):
            print(f"\nTurn {turn}: Continuing from previous context")

            response = self.send_request(
                current_context, max_tokens=200, temperature=0.1
            )
            cached_tokens = response["meta_info"]["cached_tokens"]

            print(f"Turn {turn} cached tokens: {cached_tokens}")
            print(f"Improvement: {cached_tokens - previous_cached_tokens} tokens")

            # Assert cache improvement
            self.assertGreater(
                cached_tokens,
                previous_cached_tokens,
                f"Turn {turn} should have more cached tokens than turn {turn-1}",
            )

            # Update context and cached tokens for next iteration
            current_context += response["text"]
            previous_cached_tokens = cached_tokens

            # Flush prefill cache
            self.trigger_offloading_and_flush()


if __name__ == "__main__":
    unittest.main()
