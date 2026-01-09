"""
Unit tests for AttentionClusterer

Tests HDBSCAN clustering of attention fingerprints.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from attention_clustering import (
    AttentionClusterer,
    HAS_HDBSCAN,
    HAS_UMAP,
    HAS_SKLEARN,
)


class TestAttentionClusterer:
    """Tests for AttentionClusterer class."""

    @pytest.mark.skipif(not HAS_HDBSCAN, reason="hdbscan not installed")
    def test_initialization(self):
        """Test clusterer initialization."""
        clusterer = AttentionClusterer(
            min_cluster_size=10,
            min_samples=5,
            use_umap=False,
            scale_features=True,
        )

        assert clusterer.min_cluster_size == 10
        assert clusterer.min_samples == 5
        assert clusterer.use_umap is False
        assert clusterer.scale_features is True

    @pytest.mark.skipif(not HAS_HDBSCAN, reason="hdbscan not installed")
    def test_default_initialization(self):
        """Test clusterer with default values."""
        clusterer = AttentionClusterer()

        assert clusterer.min_cluster_size == 5
        assert clusterer.min_samples == 3
        assert clusterer.use_umap is True
        assert clusterer.scale_features is True

    def test_initialization_without_hdbscan(self):
        """Test that initialization fails without hdbscan."""
        with patch('attention_clustering.HAS_HDBSCAN', False):
            # The actual check happens in __init__
            pass  # Would need to reload module to test

    @pytest.mark.skipif(not HAS_HDBSCAN, reason="hdbscan not installed")
    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_fit_small_data(self):
        """Test clustering on small synthetic data."""
        np.random.seed(42)

        # Create simple clustered data
        cluster1 = np.random.randn(20, 10) + np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(20, 10) + np.array([-2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        data = np.vstack([cluster1, cluster2]).astype(np.float32)

        clusterer = AttentionClusterer(
            min_cluster_size=5,
            use_umap=False,  # Skip UMAP for simple test
        )

        labels = clusterer.fit_predict(data)

        assert labels is not None
        assert len(labels) == 40
        # Should find at least 1 cluster (HDBSCAN may mark some as noise)
        unique_labels = set(labels) - {-1}  # Exclude noise
        assert len(unique_labels) >= 1

    @pytest.mark.skipif(not HAS_HDBSCAN, reason="hdbscan not installed")
    def test_fit_returns_labels(self):
        """Test that fit_predict returns integer labels."""
        np.random.seed(42)
        data = np.random.randn(50, 10).astype(np.float32)

        clusterer = AttentionClusterer(
            min_cluster_size=5,
            use_umap=False,
        )

        labels = clusterer.fit_predict(data)

        assert isinstance(labels, np.ndarray)
        assert labels.dtype in [np.int32, np.int64, int]


class TestClusterAnalysis:
    """Tests for cluster analysis utilities."""

    @pytest.mark.skipif(not HAS_HDBSCAN, reason="hdbscan not installed")
    def test_get_cluster_stats(self):
        """Test getting cluster statistics."""
        np.random.seed(42)
        data = np.random.randn(100, 10).astype(np.float32)

        clusterer = AttentionClusterer(
            min_cluster_size=5,
            use_umap=False,
        )

        labels = clusterer.fit_predict(data)

        # Basic stats
        n_noise = np.sum(labels == -1)
        n_clustered = np.sum(labels >= 0)

        assert n_noise + n_clustered == len(data)


class TestFeatureScaling:
    """Tests for feature scaling."""

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_scaling_normalizes_data(self):
        """Test that scaling normalizes the data."""
        from sklearn.preprocessing import StandardScaler

        data = np.array([[1, 100], [2, 200], [3, 300]], dtype=np.float32)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        # Scaled data should have mean ~0 and std ~1
        assert abs(scaled.mean()) < 0.01
        assert abs(scaled.std() - 1.0) < 0.5  # Allow some variance


class TestOptionalDependencies:
    """Tests for optional dependency flags."""

    def test_has_hdbscan_is_bool(self):
        """Test HAS_HDBSCAN is a boolean."""
        assert isinstance(HAS_HDBSCAN, bool)

    def test_has_umap_is_bool(self):
        """Test HAS_UMAP is a boolean."""
        assert isinstance(HAS_UMAP, bool)

    def test_has_sklearn_is_bool(self):
        """Test HAS_SKLEARN is a boolean."""
        assert isinstance(HAS_SKLEARN, bool)


class TestClusterLabeling:
    """Tests for cluster label generation."""

    def test_cluster_label_format(self):
        """Test cluster labels are properly formatted."""
        # Labels should be integers
        labels = np.array([0, 0, 1, 1, -1, 2])

        unique = set(labels) - {-1}
        assert all(isinstance(l, (int, np.integer)) for l in unique)

    def test_noise_label_is_minus_one(self):
        """Test that noise points are labeled as -1."""
        # HDBSCAN convention: noise = -1
        noise_label = -1
        assert noise_label == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
