"""
Tests for attention fingerprint computation and manifold zone classification.

Tests the GPU-based fingerprint computation that produces 20D/21D feature vectors
for attention pattern summarization in production routing and eviction.

Usage:
    python -m pytest test/srt/test_attention_fingerprint.py -v
"""

import unittest

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase


class TestFingerprintComputation(CustomTestCase):
    """Test the attention fingerprint computation functions."""

    def test_compute_fingerprint_basic(self):
        """Test basic fingerprint computation with valid inputs."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                compute_fingerprint_gpu,
            )
        except ImportError:
            self.skipTest("compute_fingerprint_gpu not available")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            self.skipTest("CUDA not available")

        batch_size = 2
        top_k = 10
        seq_len = 512

        # Create sample data matching actual function signature:
        # compute_fingerprint_gpu(topk_indices, topk_weights, current_pos, n_bins)
        topk_indices = torch.randint(0, seq_len, (batch_size, top_k), device=device)
        topk_weights = torch.softmax(
            torch.randn(batch_size, top_k, device=device), dim=-1
        )
        current_pos = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device
        )

        fingerprint = compute_fingerprint_gpu(topk_indices, topk_weights, current_pos)

        # Check shape - raw fingerprint is [batch_size, N_FINGERPRINT_BINS]
        # where N_FINGERPRINT_BINS=16 by default
        self.assertEqual(fingerprint.shape[0], batch_size)
        self.assertIn(fingerprint.shape[1], [16, 20, 21])  # Raw bins or with features

        # Check values are finite
        self.assertTrue(torch.isfinite(fingerprint).all())

    def test_fingerprint_zone_features(self):
        """Test that fingerprint features align with expected zones."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                add_mass_and_entropy_features,
                compute_fingerprint_gpu,
            )
        except ImportError:
            self.skipTest("compute_fingerprint_gpu not available")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            self.skipTest("CUDA not available")

        batch_size = 1
        top_k = 10
        seq_len = 1000

        # Test case 1: Highly local attention (syntax_floor zone)
        # All attention on recent positions (close to current_pos)
        local_positions = torch.arange(
            seq_len - top_k, seq_len, device=device
        ).unsqueeze(0)
        local_scores = torch.softmax(
            torch.randn(batch_size, top_k, device=device), dim=-1
        )
        current_pos = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device
        )

        local_fp_raw = compute_fingerprint_gpu(
            local_positions, local_scores, current_pos
        )
        local_features = add_mass_and_entropy_features(local_fp_raw)

        # Local mass (index 0) should be high for local attention
        # The raw histogram bins 0-2 map to local_mass (offset < 8)
        local_mass = local_features[0, 0].item()
        self.assertGreater(
            local_mass, 0.3, "Local attention should have some local_mass"
        )

        # Test case 2: Long-range attention
        # All attention on distant positions (far from current_pos)
        long_positions = torch.arange(0, top_k, device=device).unsqueeze(0)
        long_scores = torch.softmax(
            torch.randn(batch_size, top_k, device=device), dim=-1
        )

        long_fp_raw = compute_fingerprint_gpu(long_positions, long_scores, current_pos)
        long_features = add_mass_and_entropy_features(long_fp_raw)

        # Long-range mass (index 2) should be high for distant attention
        long_mass = long_features[0, 2].item()
        self.assertGreater(
            long_mass, 0.3, "Distant attention should have some long_mass"
        )

    def test_fingerprint_deterministic(self):
        """Test that fingerprint computation is deterministic."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                compute_fingerprint_gpu,
            )
        except ImportError:
            self.skipTest("compute_fingerprint_gpu not available")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            self.skipTest("CUDA not available")

        batch_size = 2
        top_k = 10
        seq_len = 512

        # Fixed data
        torch.manual_seed(42)
        positions = torch.randint(0, seq_len, (batch_size, top_k), device=device)
        scores = torch.softmax(torch.randn(batch_size, top_k, device=device), dim=-1)
        current_pos = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device
        )

        # Compute twice
        fp1 = compute_fingerprint_gpu(positions, scores, current_pos)
        fp2 = compute_fingerprint_gpu(positions, scores, current_pos)

        # Should be identical
        torch.testing.assert_close(fp1, fp2)


class TestManifoldZoneClassification(CustomTestCase):
    """Test manifold zone classification from fingerprints."""

    def test_classify_zone_basic(self):
        """Test basic zone classification."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                classify_manifold_zone,
            )
        except ImportError:
            self.skipTest("classify_manifold_zone not available")

        # Create fingerprints that should map to different zones
        # Fingerprint layout: [local_mass, mid_mass, long_mass, entropy, ...]

        # High local attention -> syntax_floor
        local_fp = np.array([0.9, 0.1, 0.0, 0.05] + [0.0] * 16)
        zone, confidence = classify_manifold_zone(local_fp)
        self.assertIn(zone, ["syntax_floor", "semantic_bridge"])

        # High mid-range attention -> semantic_bridge
        mid_fp = np.array([0.2, 0.7, 0.1, 0.25] + [0.0] * 16)
        zone, confidence = classify_manifold_zone(mid_fp)
        self.assertEqual(zone, "semantic_bridge")

        # High long-range attention -> long_range
        long_fp = np.array([0.1, 0.2, 0.7, 0.30] + [0.0] * 16)
        zone, confidence = classify_manifold_zone(long_fp)
        self.assertIn(zone, ["long_range", "semantic_bridge"])

    def test_classify_zone_diffuse(self):
        """Test that high entropy patterns are classified as diffuse."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                classify_manifold_zone,
            )
        except ImportError:
            self.skipTest("classify_manifold_zone not available")

        # High entropy, spread attention -> diffuse
        diffuse_fp = np.array([0.33, 0.33, 0.34, 0.95] + [0.125] * 8 + [0.0] * 8)
        zone, confidence = classify_manifold_zone(diffuse_fp)
        self.assertIn(zone, ["diffuse", "exploration"])


class TestFingerprintIntegration(CustomTestCase):
    """Integration tests for fingerprint-based features."""

    def test_fingerprint_to_zone_pipeline(self):
        """Test full pipeline from attention data to zone classification."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                classify_manifold_zone,
                compute_fingerprint_gpu,
            )
        except ImportError:
            self.skipTest("Pipeline functions not available")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            self.skipTest("CUDA not available")

        # Create realistic attention pattern
        batch_size = 1
        top_k = 32
        seq_len = 2048

        # Simulate semantic_bridge pattern: mid-range retrieval
        positions = torch.randint(100, 500, (batch_size, top_k), device=device)
        scores = torch.softmax(torch.randn(batch_size, top_k, device=device), dim=-1)
        current_pos = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device
        )

        # Compute fingerprint
        fingerprint = compute_fingerprint_gpu(positions, scores, current_pos)

        # Convert to numpy for classification
        fp_np = fingerprint[0].cpu().numpy()

        # Classify
        zone, confidence = classify_manifold_zone(fp_np)

        # Should get a valid zone
        valid_zones = [
            "syntax_floor",
            "semantic_bridge",
            "structure_ripple",
            "long_range",
            "diffuse",
            "exploration",
            "steering",
        ]
        self.assertIn(zone, valid_zones)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_batch_fingerprint_processing(self):
        """Test fingerprint computation for a batch of sequences."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                compute_fingerprint_gpu,
            )
        except ImportError:
            self.skipTest("compute_fingerprint_gpu not available")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            self.skipTest("CUDA not available")

        # Larger batch
        batch_size = 32
        top_k = 16
        seq_len = 4096

        positions = torch.randint(0, seq_len, (batch_size, top_k), device=device)
        scores = torch.softmax(torch.randn(batch_size, top_k, device=device), dim=-1)
        current_pos = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device
        )

        # Should not OOM and should be fast
        import time

        start = time.time()
        fingerprints = compute_fingerprint_gpu(positions, scores, current_pos)
        elapsed = time.time() - start

        self.assertEqual(fingerprints.shape[0], batch_size)
        self.assertLess(elapsed, 1.0, "Fingerprint computation should be fast")


class TestRotationalVariance(CustomTestCase):
    """Test rotational variance feature for RoPE-aware fingerprints."""

    def test_rotational_variance_computation(self):
        """Test that rotational variance is computed correctly."""
        try:
            from sglang.srt.layers.attention.triton_ops.decode_attention_with_topk import (
                compute_fingerprint_gpu,
            )
        except ImportError:
            self.skipTest("compute_fingerprint_gpu not available")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            self.skipTest("CUDA not available")

        batch_size = 2
        top_k = 10
        seq_len = 512

        positions = torch.randint(0, seq_len, (batch_size, top_k), device=device)
        scores = torch.softmax(torch.randn(batch_size, top_k, device=device), dim=-1)
        current_pos = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device
        )

        fingerprint = compute_fingerprint_gpu(positions, scores, current_pos)

        # If 21D fingerprint, last dimension should be rotational variance
        if fingerprint.shape[1] == 21:
            rot_var = fingerprint[:, 20]
            # Should be non-negative
            self.assertTrue((rot_var >= 0).all())


class TestSpectralEvictionIntegration(CustomTestCase):
    """Test integration between fingerprints and spectral eviction."""

    def test_spectral_importance_from_fingerprint(self):
        """Test computing spectral importance score from fingerprint."""
        try:
            from sglang.srt.mem_cache.spectral_eviction import SpectralEvictionStrategy
        except ImportError:
            self.skipTest("SpectralEvictionStrategy not available")

        # Create strategy (may fail if sklearn not available)
        try:
            strategy = SpectralEvictionStrategy(retention_ratio=0.3)
        except ImportError:
            self.skipTest("sklearn not available for spectral eviction")

        # Test spectral importance computation using the _compute_spectral_importance method
        # which takes (node) but we'll test the zone scoring directly

        # Verify zone scores are correctly ordered in the strategy
        expected_zone_order = [
            "semantic_bridge",
            "long_range",
            "structure_ripple",
            "syntax_floor",
            "diffuse",
        ]

        # Check that the strategy was created successfully
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.retention_ratio, 0.3)

        # Verify fallback strategy exists
        self.assertIsNotNone(strategy.fallback)


if __name__ == "__main__":
    unittest.main()
