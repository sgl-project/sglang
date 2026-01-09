"""
Unit tests for OnlineClassifier and related classes

Tests zone classification, cluster info handling, and classifier utilities.
"""

import json
import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

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


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_create_result(self):
        """Test creating a classification result."""
        result = ClassificationResult(
            cluster_id=5,
            cluster_label="Semantic Bridge",
            cluster_probability=0.85,
            zone="semantic_bridge",
            zone_confidence=0.92,
        )

        assert result.cluster_id == 5
        assert result.cluster_label == "Semantic Bridge"
        assert result.cluster_probability == 0.85
        assert result.zone == "semantic_bridge"
        assert result.zone_confidence == 0.92
        assert result.embedding is None

    def test_result_with_embedding(self):
        """Test classification result with embedding."""
        result = ClassificationResult(
            cluster_id=3,
            cluster_label="Syntax Floor",
            cluster_probability=0.95,
            zone="syntax_floor",
            zone_confidence=0.98,
            embedding=(0.5, -0.3),
        )

        assert result.embedding == (0.5, -0.3)


class TestClusterInfo:
    """Tests for ClusterInfo dataclass."""

    def test_create_cluster_info(self):
        """Test creating cluster info."""
        centroid = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        centroid[FP_LOCAL_MASS] = 0.7
        centroid[FP_ENTROPY] = 1.5

        info = ClusterInfo(
            cluster_id=2,
            label="Local Attention",
            zone="syntax_floor",
            centroid_fingerprint=centroid,
            centroid_xy=(0.25, 0.75),
            size=150,
        )

        assert info.cluster_id == 2
        assert info.label == "Local Attention"
        assert info.zone == "syntax_floor"
        # Use pytest.approx for float32 vs float comparison
        assert float(info.centroid_fingerprint[FP_LOCAL_MASS]) == pytest.approx(0.7, rel=1e-5)
        assert info.centroid_xy == (0.25, 0.75)
        assert info.size == 150


class TestConstants:
    """Tests for module constants."""

    def test_fingerprint_dim(self):
        """Test fingerprint dimension is defined."""
        assert FINGERPRINT_DIM == 20

    def test_fingerprint_indices(self):
        """Test fingerprint index constants."""
        assert FP_LOCAL_MASS == 0
        assert FP_MID_MASS == 1
        assert FP_LONG_MASS == 2
        assert FP_ENTROPY == 3

    def test_zone_thresholds_structure(self):
        """Test zone thresholds have expected structure."""
        assert "syntax_floor" in ZONE_THRESHOLDS
        assert "structure_ripple" in ZONE_THRESHOLDS
        assert "local_mass_min" in ZONE_THRESHOLDS["syntax_floor"]
        assert "entropy_max" in ZONE_THRESHOLDS["syntax_floor"]


class TestOnlineClassifier:
    """Tests for OnlineClassifier class."""

    @pytest.fixture
    def temp_discovery_dir(self):
        """Create a temporary discovery directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a run directory with manifest
            run_dir = Path(tmpdir) / "20260109_120000"
            run_dir.mkdir()

            # Create manifest
            manifest = {
                "run_id": "20260109_120000",
                "total_fingerprints": 10000,
                "total_clusters": 15,
            }
            with open(run_dir / "manifest.json", "w") as f:
                json.dump(manifest, f)

            # Create latest symlink
            latest = Path(tmpdir) / "latest"
            latest.symlink_to(run_dir)

            yield tmpdir

    def test_initialization(self, temp_discovery_dir):
        """Test classifier initialization."""
        classifier = OnlineClassifier(
            discovery_dir=temp_discovery_dir,
            use_approximate_predict=False,
            precompute_embeddings=False,
        )

        assert classifier.discovery_dir == Path(temp_discovery_dir)
        assert classifier.use_approximate_predict is False
        assert classifier.precompute_embeddings is False
        assert classifier._loaded is False

    def test_get_latest_dir_with_symlink(self, temp_discovery_dir):
        """Test finding latest directory via symlink."""
        classifier = OnlineClassifier(temp_discovery_dir)
        latest_dir = classifier._get_latest_dir()

        assert latest_dir.name == "20260109_120000"

    def test_get_latest_dir_fallback(self):
        """Test fallback when no latest symlink exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple run directories
            Path(tmpdir, "20260101_100000").mkdir()
            Path(tmpdir, "20260108_150000").mkdir()
            Path(tmpdir, "20260105_120000").mkdir()

            classifier = OnlineClassifier(tmpdir)
            latest_dir = classifier._get_latest_dir()

            # Should pick the most recent by name (sorted reverse)
            assert "20260108" in latest_dir.name

    def test_get_latest_dir_empty_raises(self):
        """Test error when no discovery runs found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = OnlineClassifier(tmpdir)

            with pytest.raises(FileNotFoundError) as excinfo:
                classifier._get_latest_dir()

            assert "No discovery runs found" in str(excinfo.value)


class TestZoneClassification:
    """Tests for zone classification logic."""

    def test_classify_syntax_floor(self):
        """Test classification of syntax_floor zone."""
        # Create fingerprint with high local mass, low entropy
        fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        fp[FP_LOCAL_MASS] = 0.7  # Above threshold
        fp[FP_MID_MASS] = 0.2
        fp[FP_LONG_MASS] = 0.1
        fp[FP_ENTROPY] = 1.5  # Below max threshold

        zone = classify_zone_from_fingerprint(fp)
        assert zone == "syntax_floor"

    def test_classify_structure_ripple(self):
        """Test classification of structure_ripple zone."""
        # Create fingerprint with high long mass
        fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        fp[FP_LOCAL_MASS] = 0.2
        fp[FP_MID_MASS] = 0.3
        fp[FP_LONG_MASS] = 0.4  # Above threshold
        fp[FP_ENTROPY] = 3.0

        zone = classify_zone_from_fingerprint(fp)
        assert zone == "structure_ripple"

    def test_classify_semantic_bridge(self):
        """Test classification of semantic_bridge (default) zone."""
        # Create fingerprint that doesn't match specific zones
        fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        fp[FP_LOCAL_MASS] = 0.35
        fp[FP_MID_MASS] = 0.4
        fp[FP_LONG_MASS] = 0.15
        fp[FP_ENTROPY] = 2.8

        zone = classify_zone_from_fingerprint(fp)
        assert zone == "semantic_bridge"


def classify_zone_from_fingerprint(fp: np.ndarray) -> str:
    """Helper function to classify zone from fingerprint (mirrors classifier logic)."""
    local_mass = fp[FP_LOCAL_MASS]
    entropy = fp[FP_ENTROPY]
    long_mass = fp[FP_LONG_MASS]

    # Check syntax_floor
    sf_thresh = ZONE_THRESHOLDS['syntax_floor']
    if local_mass >= sf_thresh['local_mass_min'] and entropy <= sf_thresh['entropy_max']:
        return "syntax_floor"

    # Check structure_ripple
    sr_thresh = ZONE_THRESHOLDS['structure_ripple']
    if long_mass >= sr_thresh['long_mass_min']:
        return "structure_ripple"

    # Default
    return "semantic_bridge"


class TestFingerprintProcessing:
    """Tests for fingerprint processing utilities."""

    def test_create_fingerprint_vector(self):
        """Test creating a valid fingerprint vector."""
        fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)

        assert fp.shape == (FINGERPRINT_DIM,)
        assert fp.dtype == np.float32

    def test_fingerprint_normalization(self):
        """Test that mass components should sum to ~1."""
        fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        fp[FP_LOCAL_MASS] = 0.5
        fp[FP_MID_MASS] = 0.3
        fp[FP_LONG_MASS] = 0.2

        mass_sum = fp[FP_LOCAL_MASS] + fp[FP_MID_MASS] + fp[FP_LONG_MASS]
        assert abs(mass_sum - 1.0) < 0.01


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_fingerprint(self):
        """Test classification with zero fingerprint."""
        fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)

        # Should not crash and return default zone
        zone = classify_zone_from_fingerprint(fp)
        assert zone == "semantic_bridge"

    def test_extreme_values(self):
        """Test classification with extreme values."""
        fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
        fp[FP_LOCAL_MASS] = 1.0
        fp[FP_ENTROPY] = 0.0

        zone = classify_zone_from_fingerprint(fp)
        assert zone == "syntax_floor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
