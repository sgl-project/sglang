"""
Tests for advanced attention visualization features.

Tests the following experimental features:
1. Sketch Mode: Per-layer summary sketches (top_hubs, dist_hist, entropy)
2. Think Segmentation: Phase tracking for reasoning models
3. Clustering Sidecar: Online clustering infrastructure

Usage:
    python -m pytest test/srt/test_attention_features.py -v
"""

import math
import time
import unittest
from typing import Dict, List

import numpy as np

from sglang.test.test_utils import CustomTestCase


class TestAttentionSketch(CustomTestCase):
    """Test the attention sketch computation functions."""

    def test_compute_attention_sketch_basic(self):
        """Test basic sketch computation with valid inputs."""
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            compute_attention_sketch,
            SKETCH_TOP_HUBS_K,
            SKETCH_DIST_BINS,
        )

        # Sample attention data
        token_positions = [10, 50, 100, 200, 500]
        attention_scores = [0.4, 0.25, 0.15, 0.12, 0.08]
        topk_logits = [2.5, 2.0, 1.5, 1.2, 0.8]
        logsumexp = 3.0  # Log normalizer
        current_pos = 512

        sketch = compute_attention_sketch(
            token_positions, attention_scores, topk_logits, logsumexp, current_pos
        )

        # Check structure
        self.assertIn("top_hubs", sketch)
        self.assertIn("dist_hist", sketch)
        self.assertIn("entropy", sketch)
        self.assertIn("mass_captured", sketch)

        # Check top_hubs format
        self.assertIsInstance(sketch["top_hubs"], list)
        self.assertLessEqual(len(sketch["top_hubs"]), SKETCH_TOP_HUBS_K)
        if sketch["top_hubs"]:
            hub = sketch["top_hubs"][0]
            self.assertIsInstance(hub, (list, tuple))
            self.assertEqual(len(hub), 2)  # (position, score)

        # Check dist_hist
        self.assertEqual(len(sketch["dist_hist"]), SKETCH_DIST_BINS)
        self.assertTrue(all(isinstance(x, float) for x in sketch["dist_hist"]))

        # Check entropy is non-negative
        self.assertGreaterEqual(sketch["entropy"], 0)

        # Check mass_captured is non-negative (can exceed 1.0 with approximate logsumexp)
        self.assertGreaterEqual(sketch["mass_captured"], 0)

    def test_compute_attention_sketch_empty(self):
        """Test sketch computation with empty inputs."""
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            compute_attention_sketch,
            SKETCH_DIST_BINS,
        )

        sketch = compute_attention_sketch([], [], None, None, 100)

        self.assertEqual(sketch["top_hubs"], [])
        self.assertEqual(sketch["dist_hist"], [0.0] * SKETCH_DIST_BINS)
        self.assertEqual(sketch["entropy"], 0.0)
        self.assertEqual(sketch["mass_captured"], 0.0)

    def test_compute_attention_sketch_distance_binning(self):
        """Test that distance histogram correctly bins by log distance."""
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            compute_attention_sketch,
        )

        # All attention on position 0 (distance = 100 from current_pos 100)
        # log2(100) ≈ 6.64, should be in bin 6
        current_pos = 100
        token_positions = [0]
        attention_scores = [1.0]

        sketch = compute_attention_sketch(
            token_positions, attention_scores, None, None, current_pos
        )

        # Find which bin has the mass
        non_zero_bins = [i for i, v in enumerate(sketch["dist_hist"]) if v > 0.5]
        self.assertIn(6, non_zero_bins)  # log2(100) ≈ 6.64

    def test_compute_attention_sketch_entropy(self):
        """Test entropy calculation for different distributions."""
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            compute_attention_sketch,
        )

        # Uniform distribution (high entropy)
        uniform_scores = [0.2, 0.2, 0.2, 0.2, 0.2]
        uniform_positions = [10, 20, 30, 40, 50]

        sketch_uniform = compute_attention_sketch(
            uniform_positions, uniform_scores, None, None, 100
        )

        # Concentrated distribution (low entropy)
        concentrated_scores = [0.9, 0.05, 0.025, 0.015, 0.01]
        concentrated_positions = [10, 20, 30, 40, 50]

        sketch_concentrated = compute_attention_sketch(
            concentrated_positions, concentrated_scores, None, None, 100
        )

        # Uniform should have higher entropy
        self.assertGreater(sketch_uniform["entropy"], sketch_concentrated["entropy"])

    def test_merge_sketches(self):
        """Test merging multiple sketches."""
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            compute_attention_sketch,
            merge_sketches,
            SKETCH_DIST_BINS,
        )

        # Create two sketches
        sketch1 = compute_attention_sketch(
            [10, 20], [0.6, 0.4], [1.0, 0.5], 1.5, 100
        )
        sketch2 = compute_attention_sketch(
            [10, 30], [0.7, 0.3], [1.2, 0.3], 1.6, 150
        )

        merged = merge_sketches([sketch1, sketch2])

        # Check structure
        self.assertIn("top_hubs", merged)
        self.assertIn("dist_hist", merged)
        self.assertIn("avg_entropy", merged)
        self.assertIn("avg_mass_captured", merged)
        self.assertIn("num_steps", merged)

        # Check aggregation
        self.assertEqual(merged["num_steps"], 2)
        self.assertEqual(len(merged["dist_hist"]), SKETCH_DIST_BINS)

        # Position 10 should be a hub in merged result (appears in both)
        hub_positions = [pos for pos, _ in merged["top_hubs"]]
        self.assertIn(10, hub_positions)

    def test_merge_sketches_empty(self):
        """Test merging empty list of sketches."""
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            merge_sketches,
            SKETCH_DIST_BINS,
        )

        merged = merge_sketches([])

        self.assertEqual(merged["top_hubs"], [])
        self.assertEqual(merged["dist_hist"], [0.0] * SKETCH_DIST_BINS)
        self.assertEqual(merged["num_steps"], 0)


class TestOnlineMicroCluster(CustomTestCase):
    """Test the online micro-cluster for streaming clustering."""

    def test_micro_cluster_creation(self):
        """Test creating a micro-cluster."""
        from examples.attention_explorer.rapids_sidecar import OnlineMicroCluster

        point = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mc = OnlineMicroCluster(point, cluster_id=0)

        self.assertEqual(mc.cluster_id, 0)
        self.assertEqual(mc.n_points, 1)
        self.assertAlmostEqual(mc.weight, 1.0)
        np.testing.assert_array_almost_equal(mc.centroid, point)

    def test_micro_cluster_add_point(self):
        """Test adding points to a micro-cluster."""
        from examples.attention_explorer.rapids_sidecar import OnlineMicroCluster

        point1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        mc = OnlineMicroCluster(point1, cluster_id=0, decay=1.0)  # No decay for testing

        point2 = np.array([3.0, 0.0, 0.0], dtype=np.float32)
        mc.add_point(point2)

        # Centroid should be between the two points
        self.assertEqual(mc.n_points, 2)
        self.assertAlmostEqual(mc.centroid[0], 2.0, places=1)  # Average of 1 and 3

    def test_micro_cluster_distance(self):
        """Test distance calculation."""
        from examples.attention_explorer.rapids_sidecar import OnlineMicroCluster

        point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        mc = OnlineMicroCluster(point, cluster_id=0)

        test_point = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        dist = mc.distance(test_point)

        self.assertAlmostEqual(dist, 5.0, places=5)  # 3-4-5 triangle

    def test_micro_cluster_decay(self):
        """Test that decay reduces weight over time."""
        from examples.attention_explorer.rapids_sidecar import OnlineMicroCluster

        point = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        mc = OnlineMicroCluster(point, cluster_id=0, decay=0.5)

        # Add second point
        mc.add_point(np.array([2.0, 2.0, 2.0], dtype=np.float32))

        # Weight should be: 1.0 * 0.5 + 1.0 = 1.5 (old decayed + new)
        self.assertAlmostEqual(mc.weight, 1.5, places=5)

    def test_micro_cluster_staleness(self):
        """Test stale cluster detection."""
        from examples.attention_explorer.rapids_sidecar import OnlineMicroCluster

        point = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        mc = OnlineMicroCluster(point, cluster_id=0)

        # Should not be stale immediately
        self.assertFalse(mc.is_stale(max_age=300))

        # Force stale by setting last_update to past
        mc.last_update = time.time() - 400
        self.assertTrue(mc.is_stale(max_age=300))


class TestRAPIDSSidecar(CustomTestCase):
    """Test the RAPIDS sidecar clustering."""

    def test_sidecar_creation(self):
        """Test creating a sidecar instance."""
        from examples.attention_explorer.rapids_sidecar import RAPIDSSidecar

        sidecar = RAPIDSSidecar(
            buffer_size=100,
            min_cluster_size=5,
            recluster_interval=10.0,
        )

        self.assertEqual(sidecar.buffer_size, 100)
        self.assertEqual(sidecar.min_cluster_size, 5)

    def test_sidecar_add_fingerprint(self):
        """Test adding fingerprints to the sidecar."""
        from examples.attention_explorer.rapids_sidecar import RAPIDSSidecar

        sidecar = RAPIDSSidecar(buffer_size=100)

        # Add some fingerprints
        for i in range(10):
            sidecar.add_fingerprint(
                request_id=f"req-{i}",
                vector=[float(i)] * 20,
                metadata={"step": i},
            )

        self.assertEqual(len(sidecar.fingerprints), 10)

    def test_sidecar_online_mode(self):
        """Test online clustering mode."""
        from examples.attention_explorer.rapids_sidecar import RAPIDSSidecar

        sidecar = RAPIDSSidecar(
            buffer_size=100,
            online_mode=True,
            online_threshold=1.0,
            online_max_clusters=10,
        )

        # Add fingerprints that should form clusters
        # Cluster 1: around [0, 0, 0, ...]
        for i in range(5):
            sidecar.add_fingerprint(
                request_id=f"cluster1-{i}",
                vector=[0.1 * i] * 20,
            )

        # Cluster 2: around [10, 10, 10, ...]
        for i in range(5):
            sidecar.add_fingerprint(
                request_id=f"cluster2-{i}",
                vector=[10.0 + 0.1 * i] * 20,
            )

        # Should have created some clusters
        centroids = sidecar.get_centroids()
        self.assertGreater(len(centroids), 0)

    def test_sidecar_predict_cluster(self):
        """Test cluster prediction."""
        from examples.attention_explorer.rapids_sidecar import RAPIDSSidecar

        sidecar = RAPIDSSidecar(online_mode=True, online_threshold=2.0)

        # Add a point to create a cluster
        sidecar.add_fingerprint("req-1", [0.0] * 20)

        # Predict for nearby point
        cluster_id, distance = sidecar.predict_cluster([0.1] * 20)

        self.assertIsInstance(cluster_id, (int, np.integer))
        self.assertTrue(isinstance(distance, (float, np.floating)))

    def test_sidecar_stats(self):
        """Test getting sidecar statistics."""
        from examples.attention_explorer.rapids_sidecar import RAPIDSSidecar

        sidecar = RAPIDSSidecar(buffer_size=100, zmq_bind=None)

        # Add some data
        for i in range(5):
            sidecar.add_fingerprint(f"req-{i}", [float(i)] * 20)

        stats = sidecar.get_stats()

        self.assertIn("backend", stats)
        self.assertIn("buffer_size", stats)
        self.assertIn("buffer_capacity", stats)
        self.assertIn("n_clusters", stats)
        self.assertIn("zmq_enabled", stats)
        self.assertIn("zmq_received", stats)

        self.assertEqual(stats["buffer_size"], 5)
        self.assertEqual(stats["buffer_capacity"], 100)
        self.assertFalse(stats["zmq_enabled"])


class TestThinkSegmentation(CustomTestCase):
    """Test think phase segmentation for reasoning models."""

    def _make_sampling_params(self):
        """Create minimal sampling params for testing."""
        from sglang.srt.sampling.sampling_params import SamplingParams
        return SamplingParams(max_new_tokens=10)

    def test_think_phase_default(self):
        """Test that default think phase is 'output'."""
        from sglang.srt.managers.schedule_batch import Req

        req = Req(
            rid="test-1",
            origin_input_text="test",
            origin_input_ids=[1, 2, 3],
            sampling_params=self._make_sampling_params(),
        )

        self.assertEqual(req.attention_think_phase, "output")
        self.assertEqual(req.attention_think_boundary, -1)

    def test_think_phase_attributes_exist(self):
        """Test that think phase attributes are properly initialized."""
        from sglang.srt.managers.schedule_batch import Req

        req = Req(
            rid="test-1",
            origin_input_text="test",
            origin_input_ids=[1, 2, 3],
            sampling_params=self._make_sampling_params(),
            return_attention_tokens=True,
        )

        # These attributes should exist
        self.assertTrue(hasattr(req, "attention_think_phase"))
        self.assertTrue(hasattr(req, "attention_think_boundary"))

        # Verify types
        self.assertIsInstance(req.attention_think_phase, str)
        self.assertIsInstance(req.attention_think_boundary, int)


class TestClusterTraitInterpretation(CustomTestCase):
    """Test cluster centroid trait interpretation."""

    def test_interpret_local_attention(self):
        """Test interpretation of local attention pattern."""
        from examples.attention_explorer.rapids_sidecar import RAPIDSSidecar

        sidecar = RAPIDSSidecar()

        # High local_mass (first element)
        centroid = np.array([0.8, 0.1, 0.05, 0.3] + [0.0] * 16)
        traits = sidecar._interpret_centroid(centroid)

        self.assertIn("syntax_floor", traits)
        self.assertIn("local_attention", traits)

    def test_interpret_semantic_bridge(self):
        """Test interpretation of semantic bridge pattern."""
        from examples.attention_explorer.rapids_sidecar import RAPIDSSidecar

        sidecar = RAPIDSSidecar()

        # High mid_mass (second element)
        centroid = np.array([0.2, 0.5, 0.1, 0.4] + [0.0] * 16)
        traits = sidecar._interpret_centroid(centroid)

        self.assertIn("semantic_bridge", traits)
        self.assertIn("retrieval_heavy", traits)

    def test_interpret_long_range(self):
        """Test interpretation of long-range attention pattern."""
        from examples.attention_explorer.rapids_sidecar import RAPIDSSidecar

        sidecar = RAPIDSSidecar()

        # High long_mass (third element)
        centroid = np.array([0.1, 0.1, 0.5, 0.4] + [0.0] * 16)
        traits = sidecar._interpret_centroid(centroid)

        self.assertIn("long_range", traits)
        self.assertIn("context_aware", traits)

    def test_interpret_focused_attention(self):
        """Test interpretation of focused attention pattern."""
        from examples.attention_explorer.rapids_sidecar import RAPIDSSidecar

        sidecar = RAPIDSSidecar()

        # Low entropy (fourth element)
        centroid = np.array([0.3, 0.3, 0.1, 0.3] + [0.0] * 16)
        traits = sidecar._interpret_centroid(centroid)

        self.assertIn("focused", traits)

    def test_get_sampling_hint(self):
        """Test sampling hint generation from traits."""
        from examples.attention_explorer.rapids_sidecar import RAPIDSSidecar

        sidecar = RAPIDSSidecar()

        # Test syntax_floor trait -> low temperature
        hints = sidecar._get_sampling_hint(["syntax_floor", "focused"])
        self.assertLess(hints["temperature"], 0.5)

        # Test diffuse trait -> higher temperature
        hints = sidecar._get_sampling_hint(["diffuse"])
        self.assertGreater(hints["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
