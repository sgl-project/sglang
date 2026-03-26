"""
Usage:
cd test/registered/core
python3 -m unittest test_flush_mm_cache.TestFlushMMEmbeddingCache
python3 -m unittest test_flush_mm_cache.TestSchedulerFlushCache
python3 -m unittest test_flush_mm_cache.TestFlushCacheEndpoint
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import requests
import torch

import sglang.srt.managers.mm_utils as mm_mod
from sglang.srt.managers.mm_utils import (
    flush_mm_embedding_cache,
    init_mm_embedding_cache,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=30, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=30, suite="stage-b-test-small-1-gpu-amd")


class TestFlushMMEmbeddingCache(unittest.TestCase):
    def tearDown(self):
        mm_mod.embedding_cache = None

    def test_returns_false_when_uninitialized(self):
        """Verify flush returns False when the cache has not been initialized."""
        mm_mod.embedding_cache = None
        self.assertFalse(flush_mm_embedding_cache())

    def test_returns_true_and_clears_cache(self):
        """Verify flush returns True and empties the cache."""
        init_mm_embedding_cache(max_size=1024)
        dummy = EmbeddingResult(embedding=torch.zeros(4))
        mm_mod.embedding_cache.set(12345, dummy)
        self.assertEqual(len(mm_mod.embedding_cache), 1)
        self.assertGreater(mm_mod.embedding_cache.current_size, 0)

        result = flush_mm_embedding_cache()

        self.assertTrue(result)
        self.assertEqual(len(mm_mod.embedding_cache), 0)
        self.assertEqual(mm_mod.embedding_cache.current_size, 0)

    def test_idempotent(self):
        """Verify consecutive flushes succeed without error."""
        init_mm_embedding_cache(max_size=1024)
        dummy = EmbeddingResult(embedding=torch.zeros(4))
        mm_mod.embedding_cache.set(12345, dummy)

        self.assertTrue(flush_mm_embedding_cache())
        self.assertTrue(flush_mm_embedding_cache())
        self.assertEqual(len(mm_mod.embedding_cache), 0)


class TestSchedulerFlushCache(unittest.TestCase):
    @patch("sglang.srt.managers.scheduler.torch")
    @patch("sglang.srt.managers.scheduler.flush_mm_embedding_cache")
    def test_happy_path_calls_flush_mm(self, mock_flush_mm, mock_torch):
        """Verify flush_cache() calls flush_mm_embedding_cache() when idle."""
        scheduler = MagicMock()
        scheduler._is_no_request.return_value = True
        scheduler.draft_worker = None

        result = Scheduler.flush_cache(scheduler)

        self.assertTrue(result)
        mock_flush_mm.assert_called_once()
        scheduler.tree_cache.reset.assert_called_once()
        scheduler.req_to_token_pool.clear.assert_called_once()
        scheduler.token_to_kv_pool_allocator.clear.assert_called_once()
        scheduler.grammar_manager.clear.assert_called_once()
        scheduler.reset_metrics.assert_called_once()

    @patch("sglang.srt.managers.scheduler.torch")
    @patch("sglang.srt.managers.scheduler.flush_mm_embedding_cache")
    def test_blocked_path_skips_flush_mm(self, mock_flush_mm, mock_torch):
        """Verify flush_cache() skips flush when requests are pending."""
        scheduler = MagicMock()
        scheduler._is_no_request.return_value = False
        scheduler.waiting_queue = ["fake_req"]
        scheduler.running_batch.reqs = ["fake_req"]

        result = Scheduler.flush_cache(scheduler)

        self.assertFalse(result)
        mock_flush_mm.assert_not_called()
        scheduler.tree_cache.reset.assert_not_called()


class TestFlushCacheEndpoint(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_flush_cache_success(self):
        """Verify 200 status and MM embedding cache mentioned in response."""
        time.sleep(1)
        response = requests.post(self.base_url + "/flush_cache")

        self.assertEqual(response.status_code, 200)
        self.assertIn("MM embedding cache", response.text)

    def test_flush_cache_response_via_get(self):
        """Verify endpoint accepts GET as well as POST."""
        response = requests.get(self.base_url + "/flush_cache")

        self.assertEqual(response.status_code, 200)
        self.assertIn("MM embedding cache", response.text)

    def test_flush_cache_blocked_by_running_request(self):
        """Verify 400 when requests are in flight."""

        def long_generate():
            return requests.post(
                self.base_url + "/generate",
                json={
                    "text": "Write a very long essay about the history of science.",
                    "sampling_params": {"max_new_tokens": 512},
                },
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(long_generate)
            time.sleep(1)
            response = requests.post(self.base_url + "/flush_cache")
            future.result()

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
