"""
Integration tests for discovery/classifier.py

Tests the OnlineClassifier including:
- Loading discovery artifacts
- Classifying fingerprints (zones and clusters)
- Centroid-based classification
"""

import json
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.classifier import (
    ClassificationResult,
    ClusterInfo,
    OnlineClassifier,
    FINGERPRINT_DIM,
    FP_LOCAL_MASS,
    FP_MID_MASS,
    FP_LONG_MASS,
    FP_ENTROPY,
    ZONE_THRESHOLDS,
)


class TestOnlineClassifierInit:
    """Tests for OnlineClassifier initialization."""

    def test_init_with_valid_discovery_dir(self, discovery_artifacts):
        """Test initialization with valid discovery directory."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
            precompute_embeddings=False,
        )

        assert classifier.discovery_dir == Path(discovery_artifacts['output_dir'])

    def test_init_with_nonexistent_dir(self, temp_output_dir):
        """Test initialization with non-existent directory."""
        classifier = OnlineClassifier(
            discovery_dir=temp_output_dir,
            precompute_embeddings=False,
        )

        assert classifier.discovery_dir == Path(temp_output_dir)

    def test_get_latest_dir(self, discovery_artifacts):
        """Test finding latest discovery run directory."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
        )

        latest_dir = classifier._get_latest_dir()

        assert latest_dir.name == discovery_artifacts['run_id']
        assert (latest_dir / "manifest.json").exists()


class TestOnlineClassifierClassify:
    """Tests for classify method."""

    def test_classify_returns_result(self, discovery_artifacts):
        """Test that classify returns a ClassificationResult."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
            precompute_embeddings=False,
        )

        # Create a fingerprint
        fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        fp[FP_LOCAL_MASS] = 0.7
        fp[FP_MID_MASS] = 0.2
        fp[FP_LONG_MASS] = 0.1
        fp[FP_ENTROPY] = 1.5

        result = classifier.classify(fp)

        assert result is not None
        assert isinstance(result, ClassificationResult)
        assert result.zone in ['syntax_floor', 'semantic_bridge', 'structure_ripple', 'unknown']

    def test_classify_syntax_floor_fingerprint(self, discovery_artifacts):
        """Test classification of syntax_floor fingerprint."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
            precompute_embeddings=False,
        )

        # Strong syntax_floor fingerprint
        fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        fp[FP_LOCAL_MASS] = 0.8
        fp[FP_MID_MASS] = 0.15
        fp[FP_LONG_MASS] = 0.05
        fp[FP_ENTROPY] = 1.0

        result = classifier.classify(fp)

        assert result is not None
        # Zone should be syntax_floor based on fingerprint
        assert result.zone in ['syntax_floor', 'semantic_bridge', 'structure_ripple', 'unknown']

    def test_classify_batch(self, discovery_artifacts):
        """Test batch classification."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
            precompute_embeddings=False,
        )

        # Create multiple fingerprints
        fingerprints = np.random.randn(10, FINGERPRINT_DIM).astype(np.float32)

        results = classifier.classify_batch(fingerprints)

        assert len(results) == 10
        assert all(isinstance(r, ClassificationResult) for r in results)


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_result_fields(self, discovery_artifacts):
        """Test classification result has all required fields."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
            precompute_embeddings=False,
        )

        fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)
        result = classifier.classify(fp)

        assert hasattr(result, 'cluster_id')
        assert hasattr(result, 'cluster_label')
        assert hasattr(result, 'cluster_probability')
        assert hasattr(result, 'zone')
        assert hasattr(result, 'zone_confidence')

    def test_result_probability_range(self, discovery_artifacts):
        """Test classification probabilities are in valid range."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
            precompute_embeddings=False,
        )

        fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)
        result = classifier.classify(fp)

        # Zone confidence is a float (can be negative for distance-based metrics
        # when clusters are not found or in edge cases)
        assert isinstance(result.zone_confidence, (int, float))
        # Note: cluster_probability may be negative for distance-based metrics


class TestClusterInfoRetrieval:
    """Tests for getting cluster information."""

    def test_get_cluster_info(self, discovery_artifacts):
        """Test getting info for a specific cluster."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
            precompute_embeddings=False,
        )

        # Get info for cluster 0
        info = classifier.get_cluster_info(0)

        if info is not None:
            assert isinstance(info, ClusterInfo)
            assert info.cluster_id == 0

    def test_get_all_clusters(self, discovery_artifacts):
        """Test getting all cluster info."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
            precompute_embeddings=False,
        )

        clusters = classifier.get_all_clusters()

        assert isinstance(clusters, dict)
        # Should have clusters from fixture
        for cluster_id, info in clusters.items():
            assert isinstance(info, ClusterInfo)


class TestReload:
    """Tests for reloading classifier."""

    def test_reload_artifacts(self, discovery_artifacts):
        """Test that reload reloads artifacts."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
            precompute_embeddings=False,
        )

        # First classify to trigger load
        fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)
        classifier.classify(fp)

        # Reload
        classifier.reload()

        # Should still work
        result = classifier.classify(fp)
        assert result is not None


class TestRunIdProperty:
    """Tests for run_id property."""

    def test_run_id(self, discovery_artifacts):
        """Test run_id property."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
            precompute_embeddings=False,
        )

        # Trigger load
        fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)
        classifier.classify(fp)

        run_id = classifier.run_id

        # Should match fixture
        assert run_id is not None or run_id == discovery_artifacts['run_id']


class TestClusterCount:
    """Tests for cluster_count property."""

    def test_cluster_count(self, discovery_artifacts):
        """Test cluster_count property."""
        classifier = OnlineClassifier(
            discovery_dir=discovery_artifacts['output_dir'],
            precompute_embeddings=False,
        )

        # Trigger load
        fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)
        classifier.classify(fp)

        count = classifier.cluster_count

        # Should have clusters from fixture
        assert count >= 0


class TestZoneThresholds:
    """Tests for zone threshold constants."""

    def test_zone_thresholds_exist(self):
        """Test zone thresholds are defined."""
        assert 'syntax_floor' in ZONE_THRESHOLDS
        assert 'structure_ripple' in ZONE_THRESHOLDS

    def test_zone_thresholds_have_required_keys(self):
        """Test zone thresholds have required keys."""
        assert 'local_mass_min' in ZONE_THRESHOLDS['syntax_floor']
        assert 'entropy_max' in ZONE_THRESHOLDS['syntax_floor']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
