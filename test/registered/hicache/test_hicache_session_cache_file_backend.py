"""
Benchmark tests for Session Cache with File Backend.
Usage:
    python3 -m pytest test/registered/hicache/test_hicache_session_cache_file_backend.py -v
"""

import json
import time
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from test_hicache_storage_file_backend import HiCacheStorageBaseMixin

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=200, suite="stage-b-test-large-2-gpu")
register_amd_ci(est_time=300, suite="stage-b-test-large-2-gpu-amd")


@dataclass
class SessionCacheSegment:
    token_start: int
    token_length: int
    kv_uri: str
    kv_start: int
    kv_length: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_length

    @property
    def kv_end(self) -> Optional[int]:
        return self.kv_start + self.kv_length if self.kv_length is not None else None


class SessionCache:
    def __init__(
        self,
        segments: Optional[List[Dict]] = None,
    ):
        if segments is None:
            self._segments: List[SessionCacheSegment] = []
            return

        self._segments = [SessionCacheSegment(**seg) for seg in segments]

    def to_dicts(self) -> List[Dict]:
        return [seg.to_dict() for seg in self._segments]

    def __len__(self) -> int:
        return len(self._segments)

    @property
    def token_end(self) -> Optional[int]:
        return self._segments[-1].token_end if self._segments else None

    @property
    def kv_end(self) -> Optional[int]:
        return self._segments[-1].kv_end if self._segments else None


class HiCacheSessionCacheBaseMixin(HiCacheStorageBaseMixin):
    """Base mixin for HiCache Storage session cache tests"""

    # Constants for test configuration
    DEFAULT_SESSION_ID = "test_session"
    DEFAULT_TOKEN_LENGTH = 1024
    CACHE_HIT_THRESHOLD = 800
    PROMPT_TOKEN_LENGTH = 768
    MAX_TOKENS = 150

    @classmethod
    def _get_base_server_args(cls):
        """Get base server arguments - can be extended in subclasses"""
        extra_config = {
            "prefetch_threshold": 100,
            "hicache_storage_pass_prefix_keys": True,
        }

        return {
            "--enable-hierarchical-cache": True,
            "--enable-session-cache": True,
            "--mem-fraction-static": 0.6,
            "--hicache-ratio": 1.2,
            "--page-size": 64,
            "--enable-cache-report": True,
            "--hicache-storage-backend-extra-config": json.dumps(extra_config),
        }

    @classmethod
    def _prepare_session_cache_file(cls, name: str) -> str:
        """Create and return session cache file path"""
        path = Path(f"{cls.temp_dir}/{name}")
        path.touch()
        return str(path)

    def _extract_text_from_response(self, response: Dict) -> str:
        """Extract text from response with validation"""
        text = response.get("text")
        self.assertIsNotNone(text, "Response should contain text")
        self.assertIsInstance(text, str, "text should be a string")
        return text

    def _extract_fresh_kv_cache_from_response(self, response: Dict) -> SessionCache:
        """Extract fresh kv cache from response with validation"""
        session_params = response.get("session_params")
        self.assertIsNotNone(session_params, "Response should contain session_params")
        self.assertIsInstance(session_params, dict, "session_params should be a dict")

        fresh_kv_cache = session_params.get("fresh_kv_cache")
        self.assertIsNotNone(
            fresh_kv_cache, "Session params should contain fresh_kv_cache"
        )
        self.assertIsInstance(fresh_kv_cache, list, "fresh_kv_cache should be a list")

        return SessionCache(fresh_kv_cache)

    def send_request_with_session(
        self,
        prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = 0.0,
        session_params: Optional[Dict] = None,
        timeout: int = 60,
    ) -> Dict:
        """Send a generate request and return response with proper error handling"""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
                "session_params": session_params,
            },
            timeout=timeout,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def _create_session_params(
        self, session_id: str, stored_kv_cache: Optional[SessionCache] = None
    ) -> Dict:
        if stored_kv_cache is None:
            fresh_kv_cache = [
                {
                    "token_start": 0,
                    "token_length": self.DEFAULT_TOKEN_LENGTH,
                    "kv_uri": f"file:///{session_id}",
                    "kv_start": 0,
                }
            ]
        else:
            fresh_kv_cache = [
                {
                    "token_start": stored_kv_cache.token_end,
                    "token_length": self.DEFAULT_TOKEN_LENGTH,
                    "kv_uri": f"file:///{session_id}",
                    "kv_start": stored_kv_cache.kv_end,
                }
            ]

        return {
            "id": session_id,
            "stored_kv_cache": stored_kv_cache.to_dicts() if stored_kv_cache else [],
            "fresh_kv_cache": fresh_kv_cache,
        }

    def test_basic_backup_and_prefetch(self):
        """Test storage and retrieval of large context through session cache"""
        print(
            "\n=== Testing Large Context Cache Storage & Retrieval by Session Cache ==="
        )

        # Generate substantial context that will be cached
        base_prompt = self.gen_prompt(self.PROMPT_TOKEN_LENGTH)

        # Prepare session cache file
        session_cache_file = self._prepare_session_cache_file(self.DEFAULT_SESSION_ID)
        print(f"Session cache file created: {session_cache_file}")

        # Step 1: Populate cache with initial request
        print("Step 1: Populating cache with large context...")
        session_params1 = self._create_session_params(self.DEFAULT_SESSION_ID)
        response1 = self.send_request_with_session(
            base_prompt, session_params=session_params1
        )

        self.assertIsNotNone(response1, "First response should not be None")

        # Extract data from first response
        generated_text = self._extract_text_from_response(response1)
        stored_kv_cache = self._extract_fresh_kv_cache_from_response(response1)
        self.assertGreater(
            len(stored_kv_cache),
            0,
            "Fresh kv cache should contain at least one segment",
        )

        # Flush device cache to force remote storage access
        self.trigger_offloading_and_flush()

        # Step 2: Test cache hit from session cache
        print("Step 2: Testing cache hit from session cache...")
        session_params2 = self._create_session_params(
            self.DEFAULT_SESSION_ID, stored_kv_cache=stored_kv_cache
        )

        start_time = time.time()
        response2 = self.send_request_with_session(
            base_prompt + generated_text, session_params=session_params2
        )
        retrieval_time = time.time() - start_time

        self.assertIsNotNone(response2, "Second response should not be None")

        cached_tokens = self.get_cached_tokens(response2)
        print(
            f"Remote cache retrieval time: {retrieval_time:.3f}s, cached_tokens={cached_tokens}"
        )

        # Assert cached tokens indicate a remote hit
        self.assertGreater(
            cached_tokens,
            self.CACHE_HIT_THRESHOLD,
            f"Expected at least {self.CACHE_HIT_THRESHOLD} cached tokens for remote hit, got {cached_tokens}",
        )


class TestHiCacheSessionCache(HiCacheSessionCacheBaseMixin, CustomTestCase):
    """Test class for HiCache session cache functionality"""

    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
