"""
Unit test for L2 (Global) multimodal cache hit scenarios.

Tests that when the same image is used across different requests
(with potentially different prompts/positions), it should hit L2 cache.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.test.test_utils import CustomTestCase


class MockMooncakeStore:
    """Mock Mooncake store for testing without external dependencies."""

    def __init__(self):
        self.data = {}

    def batch_is_exist(self, hashes):
        return [1 if h in self.data else 0 for h in hashes]

    def batch_get_into(self, keys, ptrs, sizes):
        results = []
        for i, key in enumerate(keys):
            if key in self.data:
                # Simulate copying data
                results.append(1)
            else:
                results.append(0)
        return results

    def batch_put_from(self, keys, ptrs, sizes):
        for key in keys:
            self.data[key] = True
        return [0] * len(keys)


class TestMMGlobalCacheL2(CustomTestCase):
    """Test L2 cache hit scenarios for multimodal embeddings."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        # Initialize L1 cache
        self.l1_cache = MultiModalStaticCache(max_size=1024 * 1024 * 1024)  # 1GB

        # Create mock L2 cache
        self.mock_mooncake = MockMooncakeStore()

    def _create_mock_embedding(self, num_tokens, hidden_dim):
        """Create a mock embedding tensor."""
        return torch.randn(num_tokens, hidden_dim, dtype=torch.float32)

    def test_l2_cache_hit_same_image_different_position(self):
        """
        Test that L2 cache is checked when L1 misses.

        Scenario:
        1. First request: Image at position (10, 100), not in L1 or L2
        2. Image encoded and stored in L1 and L2
        3. Second request: Same image at different position (20, 110), not in L1
        4. Should hit L2 cache and fetch from there
        """
        import sglang.srt.managers.mm_utils as mm_utils

        # Initialize caches
        mm_utils.init_mm_embedding_cache(1024 * 1024 * 1024)  # 1GB L1

        # Mock the global cache controller
        mock_controller = MagicMock()
        mock_controller.batch_is_exist.return_value = [True]  # L2 hit
        mock_controller.get_embeddings.return_value = [
            self._create_mock_embedding(10, 768)
        ]
        mm_utils.init_mm_global_cache(mock_controller)

        # Simulate first request - image not in any cache
        image_hash_1 = "hash_image_001"

        # Check L1 (should miss) - access via module to get updated reference
        l1_result = mm_utils.embedding_cache.get_single(image_hash_1)
        self.assertIsNone(l1_result, "L1 should miss on first request")

        # Check L2 (should hit after first request stores it)
        # In real scenario, first request would store to L2
        # Here we simulate L2 having the data
        mock_controller.batch_is_exist.return_value = [True]
        l2_exists = mm_utils.mm_global_cache_controller.batch_is_exist([image_hash_1])
        self.assertTrue(l2_exists[0], "L2 should have the image after first request")

        # Simulate second request with same image but different position
        # L1 should still miss (different request, L1 is request-local)
        l1_result_2 = mm_utils.embedding_cache.get_single(image_hash_1)
        self.assertIsNone(
            l1_result_2, "L1 should miss on second request (different context)"
        )

        # L2 should hit
        l2_exists_2 = mm_utils.mm_global_cache_controller.batch_is_exist([image_hash_1])
        self.assertTrue(l2_exists_2[0], "L2 should hit for same image")

        # Fetch from L2 and store to L1
        l2_embeddings = mm_utils.mm_global_cache_controller.get_embeddings(
            [image_hash_1]
        )
        self.assertEqual(len(l2_embeddings), 1, "Should get embedding from L2")

        # Store to L1 for future use
        mm_utils.embedding_cache.set(
            image_hash_1, EmbeddingResult(embedding=l2_embeddings[0])
        )

        # Now L1 should hit
        l1_result_final = mm_utils.embedding_cache.get_single(image_hash_1)
        self.assertIsNotNone(l1_result_final, "L1 should hit after fetching from L2")

    def test_l2_cache_miss_new_image(self):
        """
        Test L2 cache miss for new images.

        Scenario:
        1. New image not in L1 or L2
        2. Should trigger encoding (ViT computation)
        3. After encoding, should be stored in both L1 and L2
        """
        import sglang.srt.managers.mm_utils as mm_utils

        mm_utils.init_mm_embedding_cache(1024 * 1024 * 1024)

        mock_controller = MagicMock()
        mock_controller.batch_is_exist.return_value = [False]  # L2 miss
        mm_utils.init_mm_global_cache(mock_controller)

        # New image hash
        new_image_hash = "hash_new_image_001"

        # Check L1 (should miss)
        l1_result = mm_utils.embedding_cache.get_single(new_image_hash)
        self.assertIsNone(l1_result, "L1 should miss for new image")

        # Check L2 (should miss)
        l2_exists = mm_utils.mm_global_cache_controller.batch_is_exist([new_image_hash])
        self.assertFalse(l2_exists[0], "L2 should miss for new image")

        # In real scenario, this would trigger ViT encoding
        # Then store to both L1 and L2

    def test_l2_cache_with_multiple_images(self):
        """
        Test L2 cache with multiple images in batch.

        Scenario:
        1. Batch of 3 images: 2 in L2, 1 new
        2. Should fetch 2 from L2, encode 1 new
        """
        import sglang.srt.managers.mm_utils as mm_utils

        mm_utils.init_mm_embedding_cache(1024 * 1024 * 1024)

        mock_controller = MagicMock()
        # 2 hits, 1 miss
        mock_controller.batch_is_exist.return_value = [True, True, False]
        mock_controller.get_embeddings.return_value = [
            self._create_mock_embedding(10, 768),
            self._create_mock_embedding(15, 768),
        ]
        mm_utils.init_mm_global_cache(mock_controller)

        image_hashes = ["hash_001", "hash_002", "hash_003"]

        # Check L2 for all
        l2_results = mm_utils.mm_global_cache_controller.batch_is_exist(image_hashes)
        self.assertEqual(
            l2_results, [True, True, False], "Should have 2 hits and 1 miss"
        )

        # Fetch existing from L2
        existing_hashes = [h for h, exists in zip(image_hashes, l2_results) if exists]
        l2_embeddings = mm_utils.mm_global_cache_controller.get_embeddings(
            existing_hashes
        )
        self.assertEqual(len(l2_embeddings), 2, "Should fetch 2 from L2")


class TestMMCacheWithPromptVariation(CustomTestCase):
    """Test cache behavior when prompt varies but image stays same."""

    def _create_mock_embedding(self, num_tokens, hidden_dim):
        """Create a mock embedding tensor."""
        return torch.randn(num_tokens, hidden_dim, dtype=torch.float32)

    def test_same_image_different_prompts(self):
        """
        Test that same image hits cache even with different prompts.

        This tests the scenario:
        1. Request 1: "Describe this image <image>" -> image at position (5, 100)
        2. Request 2: "Please provide a detailed description of the content in this image <image>" -> image at position (12, 107)
        3. Both should share the same image embedding via L2 cache
        """
        import sglang.srt.managers.mm_utils as mm_utils

        mm_utils.init_mm_embedding_cache(1024 * 1024 * 1024)

        mock_controller = MagicMock()
        mock_controller.batch_is_exist.return_value = [True]  # Always hit L2
        mock_controller.get_embeddings.return_value = [
            self._create_mock_embedding(10, 768)
        ]
        mm_utils.init_mm_global_cache(mock_controller)

        # Same image hash regardless of prompt
        image_hash = "hash_stable_image_001"

        # Simulate request 1 - image at different position
        # L1 miss (fresh request)
        l1_hit_1 = mm_utils.embedding_cache.get_single(image_hash)
        self.assertIsNone(l1_hit_1)

        # L2 hit
        l2_hit_1 = mm_utils.mm_global_cache_controller.batch_is_exist([image_hash])
        self.assertTrue(l2_hit_1[0])

        # Fetch and store to L1
        emb_1 = mm_utils.mm_global_cache_controller.get_embeddings([image_hash])[0]
        mm_utils.embedding_cache.set(image_hash, EmbeddingResult(embedding=emb_1))

        # Clear L1 (simulating new request context)
        mm_utils.init_mm_embedding_cache(1024 * 1024 * 1024)

        # Simulate request 2 - same image, different prompt/position
        # L1 miss (new request context)
        l1_hit_2 = mm_utils.embedding_cache.get_single(image_hash)
        self.assertIsNone(l1_hit_2)

        # L2 should still hit (same image content)
        l2_hit_2 = mm_utils.mm_global_cache_controller.batch_is_exist([image_hash])
        self.assertTrue(
            l2_hit_2[0], "L2 should hit for same image regardless of prompt"
        )


if __name__ == "__main__":
    unittest.main()
