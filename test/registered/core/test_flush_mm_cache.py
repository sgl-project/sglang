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
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=30, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=30, suite="stage-b-test-small-1-gpu-amd")


# --------------------------------------------------------------------------- #
# Case 1: flush_mm_embedding_cache() unit tests
# --------------------------------------------------------------------------- #
class TestFlushMMEmbeddingCache(unittest.TestCase):
    """Direct unit tests for the flush_mm_embedding_cache() helper."""

    def tearDown(self):
        mm_mod.embedding_cache = None

    def test_returns_false_when_uninitialized(self):
        mm_mod.embedding_cache = None
        self.assertFalse(flush_mm_embedding_cache())

    def test_returns_true_and_clears_cache(self):
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
        init_mm_embedding_cache(max_size=1024)
        dummy = EmbeddingResult(embedding=torch.zeros(4))
        mm_mod.embedding_cache.set(12345, dummy)

        self.assertTrue(flush_mm_embedding_cache())
        self.assertTrue(flush_mm_embedding_cache())
        self.assertEqual(len(mm_mod.embedding_cache), 0)


# --------------------------------------------------------------------------- #
# Cases 2 & 3: Scheduler.flush_cache() with mocked internals
# --------------------------------------------------------------------------- #
class TestSchedulerFlushCache(unittest.TestCase):
    """Verify flush_cache() calls flush_mm_embedding_cache() correctly."""

    @patch("sglang.srt.managers.scheduler.torch")
    @patch("sglang.srt.managers.scheduler.flush_mm_embedding_cache")
    def test_happy_path_calls_flush_mm(self, mock_flush_mm, mock_torch):
        """When _is_no_request() is True, flush_mm_embedding_cache() is called
        and flush_cache() returns True."""
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
        """When _is_no_request() is False, flush_mm_embedding_cache() is NOT
        called and flush_cache() returns False."""
        scheduler = MagicMock()
        scheduler._is_no_request.return_value = False
        scheduler.waiting_queue = ["fake_req"]
        scheduler.running_batch.reqs = ["fake_req"]

        result = Scheduler.flush_cache(scheduler)

        self.assertFalse(result)
        mock_flush_mm.assert_not_called()
        scheduler.tree_cache.reset.assert_not_called()


# --------------------------------------------------------------------------- #
# Case 4: /flush_cache endpoint integration test
# --------------------------------------------------------------------------- #
class TestFlushCacheEndpoint(CustomTestCase):
    """Integration test for the /flush_cache HTTP endpoint."""

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
        """Verify 200 status and updated response text."""
        time.sleep(1)
        response = requests.post(self.base_url + "/flush_cache")

        self.assertEqual(response.status_code, 200)
        self.assertIn("MM embedding cache", response.text)

    def test_flush_cache_response_via_get(self):
        """Endpoint accepts both GET and POST."""
        response = requests.get(self.base_url + "/flush_cache")

        self.assertEqual(response.status_code, 200)
        self.assertIn("MM embedding cache", response.text)

    def test_flush_cache_blocked_by_running_request(self):
        """Returns 400 when requests are in flight."""

        def long_generate():
            return requests.post(
                self.base_url + "/generate",
                json={
                    "text": "Write a very long essay about the history of science.",
                    "sampling_params": {"max_new_tokens": 512},
                },
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Fire a long-running request
            future = executor.submit(long_generate)
            # Give it time to start processing
            time.sleep(1)
            # Flush while request is in flight
            response = requests.post(self.base_url + "/flush_cache")
            # Wait for the generation to finish
            future.result()

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
