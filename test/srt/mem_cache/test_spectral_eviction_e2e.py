#!/usr/bin/env python3
"""
End-to-end smoke test for spectral eviction fingerprint wiring.

This test verifies the complete flow:
  attention capture → fingerprint computation → cache node metadata → eviction priority

Tests are designed to run without a live server by simulating the data flow.

For live server testing, use: scripts/test_spectral_eviction_live.py
"""

import time
import unittest
from unittest.mock import MagicMock
from typing import Dict, List

import numpy as np


class TestSpectralEvictionE2E(unittest.TestCase):
    """End-to-end tests for spectral eviction fingerprint flow."""

    def setUp(self):
        """Set up test fixtures."""
        # Check if radix_cache is available
        try:
            from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

            self.RadixCache = RadixCache
            self.TreeNode = TreeNode
            self.radix_available = True
        except ImportError:
            self.radix_available = False

        # Check if spectral eviction is available
        try:
            from sglang.srt.mem_cache.spectral_eviction import (
                SpectralEvictionStrategy,
                SpectralSkeletonComputer,
                ZONE_IMPORTANCE,
            )

            self.SpectralEvictionStrategy = SpectralEvictionStrategy
            self.SpectralSkeletonComputer = SpectralSkeletonComputer
            self.ZONE_IMPORTANCE = ZONE_IMPORTANCE
            self.spectral_available = True
        except ImportError:
            self.spectral_available = False

    def _create_mock_attention_tokens(self, n_tokens: int = 10) -> List[Dict]:
        """
        Create mock attention_tokens data as would come from attention capture.

        This simulates the data format produced by scheduler_output_processor_mixin.py
        """
        attention_tokens = []
        zones = ["syntax_floor", "semantic_bridge", "structure_ripple"]

        for i in range(n_tokens):
            # Create a 20D fingerprint vector
            fingerprint = np.random.rand(20).tolist()
            # Vary the patterns to get different zone classifications
            fingerprint[0] = 0.6 if i % 3 == 0 else 0.2  # local_mass
            fingerprint[1] = 0.5 if i % 3 == 1 else 0.2  # mid_mass
            fingerprint[2] = 0.4 if i % 3 == 2 else 0.1  # long_mass
            fingerprint[3] = 1.5 + (i % 4) * 0.5  # entropy

            attention_tokens.append(
                {
                    "schema_version": 1,
                    "mode": "fingerprint",
                    "fingerprint": fingerprint,
                    "manifold": zones[i % 3],
                    "step": i,
                    "think_phase": "output",
                }
            )

        return attention_tokens

    def _create_mock_request(self, attention_tokens: List[Dict]) -> MagicMock:
        """Create a mock request object with attention_tokens."""
        req = MagicMock()
        req.attention_tokens = attention_tokens
        req.last_node = None  # Will be set by the test
        return req

    def test_fingerprint_extraction_from_attention_tokens(self):
        """Test that fingerprints are correctly extracted from attention_tokens format."""
        attention_tokens = self._create_mock_attention_tokens(5)

        # Extract fingerprints like cache_finished_req_with_fingerprints does
        fingerprints = []
        manifold_zones = []

        for token_info in attention_tokens:
            if isinstance(token_info, dict):
                fp = token_info.get("fingerprint")
                zone = token_info.get("manifold")
                if fp is not None:
                    fingerprints.append(fp)
                if zone is not None:
                    manifold_zones.append(zone)

        self.assertEqual(len(fingerprints), 5)
        self.assertEqual(len(manifold_zones), 5)
        self.assertEqual(len(fingerprints[0]), 20)  # 20D fingerprint
        self.assertIn(
            manifold_zones[0], ["syntax_floor", "semantic_bridge", "structure_ripple"]
        )

    def test_attach_metadata_to_mock_node(self):
        """Test attaching spectral metadata to a mock TreeNode."""
        if not self.spectral_available:
            self.skipTest("spectral_eviction not available")

        # Create a mock node
        class MockNode:
            def __init__(self):
                self.spectral_fingerprint = None
                self.manifold_zone = None
                self.spectral_coherence = None
                self.last_access_time = time.monotonic()

        node = MockNode()
        fingerprint = np.random.rand(20)

        # Attach metadata
        node.spectral_fingerprint = fingerprint
        node.manifold_zone = "semantic_bridge"
        node.spectral_coherence = 0.85

        # Verify
        self.assertIsNotNone(node.spectral_fingerprint)
        self.assertEqual(node.manifold_zone, "semantic_bridge")
        self.assertEqual(node.spectral_coherence, 0.85)

        # Test eviction priority
        strategy = self.SpectralEvictionStrategy()
        priority = strategy.get_priority(node)

        # Should return tuple (spectral_score, lru_score)
        self.assertIsInstance(priority, tuple)
        self.assertEqual(len(priority), 2)
        self.assertGreater(priority[0], 0)  # spectral score > 0

    def test_zone_importance_affects_priority(self):
        """Test that different zones get different eviction priorities."""
        if not self.spectral_available:
            self.skipTest("spectral_eviction not available")

        class MockNode:
            def __init__(self, zone):
                self.spectral_fingerprint = np.zeros(20)
                self.manifold_zone = zone
                self.spectral_coherence = 0.5
                self.last_access_time = time.monotonic()

        strategy = self.SpectralEvictionStrategy()

        # semantic_bridge should have highest importance
        bridge_node = MockNode("semantic_bridge")
        syntax_node = MockNode("syntax_floor")

        bridge_priority = strategy.get_priority(bridge_node)
        syntax_priority = strategy.get_priority(syntax_node)

        # semantic_bridge (0.95) > syntax_floor (0.30)
        self.assertGreater(bridge_priority[0], syntax_priority[0])

    def test_full_flow_simulation(self):
        """
        Simulate the full flow from attention capture to eviction decision.

        This is the key E2E test that verifies the wiring works.
        """
        if not self.spectral_available:
            self.skipTest("spectral_eviction not available")

        # Step 1: Create attention tokens (as from attention capture)
        attention_tokens = self._create_mock_attention_tokens(10)

        # Step 2: Extract fingerprints (as cache_finished_req_with_fingerprints does)
        fingerprints = []
        manifold_zones = []
        for token_info in attention_tokens:
            if isinstance(token_info, dict):
                fp = token_info.get("fingerprint")
                zone = token_info.get("manifold")
                if fp is not None:
                    fingerprints.append(np.array(fp))
                if zone is not None:
                    manifold_zones.append(zone)

        # Step 3: Create mock cache nodes and attach metadata
        class MockNode:
            def __init__(self):
                self.spectral_fingerprint = None
                self.manifold_zone = None
                self.spectral_coherence = None
                self.last_access_time = time.monotonic()

        nodes = [MockNode() for _ in range(len(fingerprints))]
        for i, node in enumerate(nodes):
            node.spectral_fingerprint = fingerprints[i]
            node.manifold_zone = manifold_zones[i]
            node.spectral_coherence = 0.5 + np.random.rand() * 0.5

        # Step 4: Compute eviction priorities
        strategy = self.SpectralEvictionStrategy()
        priorities = [strategy.get_priority(node) for node in nodes]

        # Verify all nodes got valid priorities
        for i, priority in enumerate(priorities):
            self.assertIsInstance(priority, tuple, f"Node {i} priority should be tuple")
            self.assertEqual(
                len(priority), 2, f"Node {i} priority should have 2 elements"
            )
            self.assertGreater(priority[0], 0, f"Node {i} spectral score should be > 0")

        # Verify zone-based ordering (semantic_bridge > syntax_floor)
        bridge_priorities = [
            p[0]
            for i, p in enumerate(priorities)
            if manifold_zones[i] == "semantic_bridge"
        ]
        syntax_priorities = [
            p[0]
            for i, p in enumerate(priorities)
            if manifold_zones[i] == "syntax_floor"
        ]

        if bridge_priorities and syntax_priorities:
            avg_bridge = sum(bridge_priorities) / len(bridge_priorities)
            avg_syntax = sum(syntax_priorities) / len(syntax_priorities)
            self.assertGreater(
                avg_bridge,
                avg_syntax,
                "semantic_bridge nodes should have higher avg priority than syntax_floor",
            )

    def test_skeleton_computation_e2e(self):
        """Test skeleton computation with realistic fingerprints."""
        if not self.spectral_available:
            self.skipTest("spectral_eviction not available")

        # Create fingerprints with cluster structure
        n_tokens = 100
        np.random.seed(42)

        # 3 clusters
        cluster1 = np.random.randn(30, 20) + np.array([2, 0, 0] + [0] * 17)
        cluster2 = np.random.randn(40, 20) + np.array([0, 2, 0] + [0] * 17)
        cluster3 = np.random.randn(30, 20) + np.array([0, 0, 2] + [0] * 17)
        fingerprints = np.vstack([cluster1, cluster2, cluster3])

        # Compute skeleton
        computer = self.SpectralSkeletonComputer(
            retention_ratio=0.3,
            min_samples=10,
        )
        result = computer.compute_skeleton(fingerprints, n_tokens)

        # Verify skeleton properties
        self.assertGreater(len(result.skeleton_indices), 0)
        self.assertLessEqual(len(result.skeleton_indices), int(n_tokens * 0.35))

        # Skeleton should cover all clusters
        skeleton_set = set(result.skeleton_indices)
        has_c1 = any(i in skeleton_set for i in range(0, 30))
        has_c2 = any(i in skeleton_set for i in range(30, 70))
        has_c3 = any(i in skeleton_set for i in range(70, 100))

        self.assertTrue(has_c1, "Skeleton should cover cluster 1")
        self.assertTrue(has_c2, "Skeleton should cover cluster 2")
        self.assertTrue(has_c3, "Skeleton should cover cluster 3")

    def test_fallback_without_spectral_data(self):
        """Test that nodes without spectral data fall back to LRU."""
        if not self.spectral_available:
            self.skipTest("spectral_eviction not available")

        class MockNode:
            def __init__(self):
                self.spectral_fingerprint = None  # No spectral data
                self.manifold_zone = None
                self.spectral_coherence = None
                self.last_access_time = time.monotonic()

        node = MockNode()
        strategy = self.SpectralEvictionStrategy()
        priority = strategy.get_priority(node)

        # Should return just a float (LRU priority)
        self.assertIsInstance(priority, float)


class TestFingerprintSchemaIntegration(unittest.TestCase):
    """Test fingerprint schema is correctly used across modules."""

    def test_schema_constants_available(self):
        """Test that schema constants are importable from central module."""
        try:
            from examples.attention_explorer.discovery.fingerprint_schema import (
                V1_DIM,
                V2_DIM,
                FP_ROTATIONAL_VARIANCE,
                ZONE_THRESHOLDS,
            )

            self.assertEqual(V1_DIM, 20)
            self.assertEqual(V2_DIM, 21)
            self.assertEqual(FP_ROTATIONAL_VARIANCE, 20)
            self.assertIn("syntax_floor", ZONE_THRESHOLDS)
            self.assertIn("semantic_bridge", ZONE_THRESHOLDS)
            self.assertIn("structure_ripple", ZONE_THRESHOLDS)

        except ImportError:
            self.skipTest("fingerprint_schema not available")

    def test_is_v2_detection(self):
        """Test v2 fingerprint detection."""
        try:
            from examples.attention_explorer.discovery.fingerprint_schema import (
                V1_DIM,
                V2_DIM,
                is_v2,
            )

            v1_fp = np.zeros(V1_DIM)
            v2_fp = np.zeros(V2_DIM)

            self.assertFalse(is_v2(v1_fp))
            self.assertTrue(is_v2(v2_fp))

        except ImportError:
            self.skipTest("fingerprint_schema not available")


if __name__ == "__main__":
    unittest.main()
