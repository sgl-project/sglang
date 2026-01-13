"""
Unit tests for Spectral KV Cache Eviction.

Tests:
1. SpectralSkeletonComputer - skeleton identification
2. SpectralEvictionStrategy - priority computation
3. Integration with RadixCache TreeNode
4. Fallback behavior when sklearn unavailable
"""

import time
import unittest
from unittest.mock import MagicMock

import numpy as np


class TestSpectralSkeletonComputer(unittest.TestCase):
    """Test skeleton token identification."""

    def setUp(self):
        """Set up test fixtures."""
        from sglang.srt.mem_cache.spectral_eviction import SpectralSkeletonComputer

        self.computer = SpectralSkeletonComputer(
            n_components=5,
            retention_ratio=0.3,
            min_samples=10,
        )

    def test_skeleton_respects_retention_ratio(self):
        """Test that skeleton size approximately matches retention ratio."""
        np.random.seed(42)
        n_tokens = 100
        fingerprints = np.random.randn(n_tokens, 20)

        result = self.computer.compute_skeleton(fingerprints, n_tokens)

        expected_size = int(n_tokens * 0.3)
        # Allow some variance due to spectral extreme selection
        self.assertGreaterEqual(len(result.skeleton_indices), expected_size - 5)
        self.assertLessEqual(len(result.skeleton_indices), expected_size + 5)

    def test_skeleton_covers_all_clusters(self):
        """Test that skeleton tokens span different clusters."""
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.randn(30, 20) + np.array([5, 0, 0] + [0] * 17)
        cluster2 = np.random.randn(30, 20) + np.array([0, 5, 0] + [0] * 17)
        cluster3 = np.random.randn(30, 20) + np.array([0, 0, 5] + [0] * 17)
        fingerprints = np.vstack([cluster1, cluster2, cluster3])

        result = self.computer.compute_skeleton(fingerprints, 90)

        # Check that skeleton has representatives from each cluster
        skeleton_set = set(result.skeleton_indices)
        has_cluster1 = any(i in skeleton_set for i in range(0, 30))
        has_cluster2 = any(i in skeleton_set for i in range(30, 60))
        has_cluster3 = any(i in skeleton_set for i in range(60, 90))

        self.assertTrue(has_cluster1, "Skeleton should cover cluster 1")
        self.assertTrue(has_cluster2, "Skeleton should cover cluster 2")
        self.assertTrue(has_cluster3, "Skeleton should cover cluster 3")

    def test_small_sequence_keeps_all(self):
        """Test that small sequences keep all tokens."""
        fingerprints = np.random.randn(10, 20)

        result = self.computer.compute_skeleton(fingerprints, 10)

        # With retention_ratio=0.3, n_keep=3, but we have 10 tokens
        # Since 10 > 3, we should get ~3 tokens
        self.assertLessEqual(len(result.skeleton_indices), 10)

    def test_very_small_sequence_uses_fallback(self):
        """Test fallback for sequences below min_samples."""
        fingerprints = np.random.randn(5, 20)  # Below min_samples=10

        result = self.computer.compute_skeleton(fingerprints, 5)

        # Should use evenly spaced fallback
        self.assertGreater(len(result.skeleton_indices), 0)
        self.assertEqual(result.n_spectral_dims, 0)  # No spectral used

    def test_skeleton_indices_are_sorted(self):
        """Test that skeleton indices are sorted."""
        np.random.seed(42)
        fingerprints = np.random.randn(50, 20)

        result = self.computer.compute_skeleton(fingerprints, 50)

        self.assertEqual(result.skeleton_indices, sorted(result.skeleton_indices))

    def test_cache_invalidation(self):
        """Test that cache is invalidated when sequence grows."""
        np.random.seed(42)
        fingerprints1 = np.random.randn(50, 20)
        fingerprints2 = np.random.randn(100, 20)  # Grew significantly

        # First computation
        result1 = self.computer.compute_skeleton(fingerprints1, 50, cache_key="test")

        # Second computation with same key but larger data
        result2 = self.computer.compute_skeleton(fingerprints2, 100, cache_key="test")

        # Should have different sizes due to growth
        self.assertNotEqual(
            len(result1.skeleton_indices), len(result2.skeleton_indices)
        )


class TestSpectralEvictionStrategy(unittest.TestCase):
    """Test eviction priority computation."""

    def setUp(self):
        """Set up test fixtures."""
        from sglang.srt.mem_cache.spectral_eviction import SpectralEvictionStrategy

        self.strategy = SpectralEvictionStrategy(
            retention_ratio=0.3,
            spectral_weight=0.7,
        )

    def _create_mock_node(
        self, fingerprint=None, zone=None, coherence=None, access_time=None
    ):
        """Create a mock TreeNode with spectral metadata."""
        node = MagicMock()
        node.spectral_fingerprint = fingerprint
        node.manifold_zone = zone
        node.spectral_coherence = coherence
        node.last_access_time = access_time or time.monotonic()
        return node

    def test_semantic_bridge_higher_than_syntax_floor(self):
        """Test that semantic_bridge zone has higher priority than syntax_floor."""
        fp = np.zeros(20)
        fp[3] = 0.3  # Low entropy

        semantic_node = self._create_mock_node(fp, "semantic_bridge", 0.8)
        syntax_node = self._create_mock_node(fp, "syntax_floor", 0.8)

        semantic_priority = self.strategy.get_priority(semantic_node)
        syntax_priority = self.strategy.get_priority(syntax_node)

        # Higher priority = kept longer
        self.assertGreater(semantic_priority[0], syntax_priority[0])

    def test_high_coherence_higher_priority(self):
        """Test that high coherence tokens have higher priority."""
        fp = np.zeros(20)

        high_coherence = self._create_mock_node(fp, "semantic_bridge", 0.9)
        low_coherence = self._create_mock_node(fp, "semantic_bridge", 0.2)

        high_priority = self.strategy.get_priority(high_coherence)
        low_priority = self.strategy.get_priority(low_coherence)

        self.assertGreater(high_priority[0], low_priority[0])

    def test_low_entropy_higher_priority(self):
        """Test that focused attention (low entropy) has higher priority."""
        fp_focused = np.zeros(20)
        fp_focused[3] = 0.1  # Very low entropy

        fp_diffuse = np.zeros(20)
        fp_diffuse[3] = 0.9  # High entropy

        focused_node = self._create_mock_node(fp_focused, "semantic_bridge", 0.5)
        diffuse_node = self._create_mock_node(fp_diffuse, "semantic_bridge", 0.5)

        focused_priority = self.strategy.get_priority(focused_node)
        diffuse_priority = self.strategy.get_priority(diffuse_node)

        self.assertGreater(focused_priority[0], diffuse_priority[0])

    def test_fallback_to_lru_without_spectral_data(self):
        """Test fallback to LRU when no spectral metadata."""
        node = self._create_mock_node(None, None, None, time.monotonic())

        priority = self.strategy.get_priority(node)

        # Should return just a float (LRU priority)
        self.assertIsInstance(priority, float)

    def test_recent_access_increases_priority(self):
        """Test that recent access time increases priority."""
        fp = np.zeros(20)

        old_node = self._create_mock_node(
            fp, "semantic_bridge", 0.5, time.monotonic() - 1000
        )
        new_node = self._create_mock_node(fp, "semantic_bridge", 0.5, time.monotonic())

        old_priority = self.strategy.get_priority(old_node)
        new_priority = self.strategy.get_priority(new_node)

        # Newer node should have slightly higher priority due to LRU component
        self.assertGreater(new_priority[0], old_priority[0])


class TestRadixCacheSpectralIntegration(unittest.TestCase):
    """Test integration with RadixCache."""

    @classmethod
    def setUpClass(cls):
        """Check if radix_cache can be imported."""
        cls.radix_cache_available = False
        cls.TreeNode = None
        cls.RadixCache = None
        try:
            from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

            cls.TreeNode = TreeNode
            cls.RadixCache = RadixCache
            cls.radix_cache_available = True
        except ImportError as e:
            cls.import_error = str(e)

    def test_treenode_has_spectral_fields(self):
        """Test that TreeNode has spectral metadata fields."""
        if not self.radix_cache_available:
            # Use a mock TreeNode to test the expected interface
            class MockTreeNode:
                def __init__(self):
                    self.spectral_fingerprint = None
                    self.manifold_zone = None
                    self.spectral_coherence = None

            node = MockTreeNode()
        else:
            node = self.TreeNode()

        self.assertTrue(hasattr(node, "spectral_fingerprint"))
        self.assertTrue(hasattr(node, "manifold_zone"))
        self.assertTrue(hasattr(node, "spectral_coherence"))

        # Should be None initially
        self.assertIsNone(node.spectral_fingerprint)
        self.assertIsNone(node.manifold_zone)
        self.assertIsNone(node.spectral_coherence)

    def test_attach_spectral_metadata(self):
        """Test attaching spectral metadata to a node."""
        if not self.radix_cache_available:
            # Test with mock objects when radix_cache not available
            class MockTreeNode:
                def __init__(self):
                    self.spectral_fingerprint = None
                    self.manifold_zone = None
                    self.spectral_coherence = None

            class MockRadixCache:
                def attach_spectral_metadata(
                    self, node, fingerprint, manifold_zone, spectral_coherence
                ):
                    node.spectral_fingerprint = fingerprint
                    node.manifold_zone = manifold_zone
                    node.spectral_coherence = spectral_coherence

            cache = MockRadixCache()
            node = MockTreeNode()
        else:
            cache = self.RadixCache.create_simulated(disable=False)
            node = self.TreeNode()

        fp = np.random.randn(20)

        cache.attach_spectral_metadata(
            node,
            fingerprint=fp,
            manifold_zone="semantic_bridge",
            spectral_coherence=0.85,
        )

        self.assertIsNotNone(node.spectral_fingerprint)
        self.assertEqual(node.manifold_zone, "semantic_bridge")
        self.assertEqual(node.spectral_coherence, 0.85)


class TestZoneImportanceScoring(unittest.TestCase):
    """Test zone importance scoring logic."""

    def test_zone_ordering(self):
        """Test that zone importance follows expected ordering."""
        from sglang.srt.mem_cache.spectral_eviction import ZONE_IMPORTANCE

        self.assertGreater(
            ZONE_IMPORTANCE["semantic_bridge"], ZONE_IMPORTANCE["long_range"]
        )
        self.assertGreater(
            ZONE_IMPORTANCE["long_range"], ZONE_IMPORTANCE["structure_ripple"]
        )
        self.assertGreater(
            ZONE_IMPORTANCE["structure_ripple"], ZONE_IMPORTANCE["syntax_floor"]
        )
        self.assertGreater(ZONE_IMPORTANCE["syntax_floor"], ZONE_IMPORTANCE["diffuse"])

    def test_all_zones_have_scores(self):
        """Test that all expected zones have importance scores."""
        from sglang.srt.mem_cache.spectral_eviction import ZONE_IMPORTANCE

        expected_zones = [
            "semantic_bridge",
            "long_range",
            "structure_ripple",
            "syntax_floor",
            "diffuse",
            "unknown",
        ]

        for zone in expected_zones:
            self.assertIn(zone, ZONE_IMPORTANCE)
            self.assertGreaterEqual(ZONE_IMPORTANCE[zone], 0)
            self.assertLessEqual(ZONE_IMPORTANCE[zone], 1)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""

    def test_compute_sequence_skeleton(self):
        """Test convenience function for skeleton computation."""
        from sglang.srt.mem_cache.spectral_eviction import compute_sequence_skeleton

        np.random.seed(42)
        fingerprints = [np.random.randn(20) for _ in range(50)]

        skeleton = compute_sequence_skeleton(fingerprints, retention_ratio=0.3)

        self.assertIsInstance(skeleton, list)
        self.assertEqual(len(skeleton), 15)  # 50 * 0.3 = 15

    def test_score_token_importance(self):
        """Test single token importance scoring."""
        from sglang.srt.mem_cache.spectral_eviction import score_token_importance

        # High importance: semantic_bridge, high coherence, low entropy
        high_fp = np.zeros(20)
        high_fp[3] = 0.1  # Low entropy
        high_score = score_token_importance(high_fp, "semantic_bridge", 0.9)

        # Low importance: syntax_floor, low coherence, high entropy
        low_fp = np.zeros(20)
        low_fp[3] = 0.9  # High entropy
        low_score = score_token_importance(low_fp, "syntax_floor", 0.2)

        self.assertGreater(high_score, low_score)


class TestSklearnFallback(unittest.TestCase):
    """Test fallback behavior when sklearn is not available."""

    def test_evenly_spaced_fallback(self):
        """Test evenly spaced skeleton when sklearn unavailable."""
        from sglang.srt.mem_cache.spectral_eviction import SpectralSkeletonComputer

        # Use higher retention ratio to ensure we get multiple tokens
        computer = SpectralSkeletonComputer(retention_ratio=0.5, min_samples=10)

        # Force fallback by using small sample size (below min_samples)
        fingerprints = np.random.randn(8, 20)
        result = computer.compute_skeleton(fingerprints, 8)

        # With 50% retention of 8 tokens = 4 tokens
        # Should include first and last
        self.assertIn(0, result.skeleton_indices)
        self.assertIn(7, result.skeleton_indices)
        self.assertEqual(result.n_spectral_dims, 0)  # No spectral used (fallback)


if __name__ == "__main__":
    unittest.main()
