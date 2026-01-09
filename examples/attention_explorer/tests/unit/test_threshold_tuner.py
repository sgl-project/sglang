"""
Unit tests for ZoneThresholdTuner

Tests adaptive zone threshold learning from probe harness feedback.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.threshold_tuner import (
    ZoneThresholdTuner,
    ZoneSample,
    DEFAULT_ZONE_THRESHOLDS,
    ZONES,
    create_threshold_tuner,
)


class TestZoneSample:
    """Tests for ZoneSample dataclass."""

    def test_create_sample(self):
        """Test creating a zone sample."""
        sample = ZoneSample(
            fingerprint_id=1,
            features={
                "local_mass": 0.7,
                "mid_mass": 0.2,
                "long_mass": 0.1,
                "entropy": 1.5,
            },
            predicted_zone="syntax_floor",
            actual_zone="syntax_floor",
            confidence=0.95,
        )

        assert sample.fingerprint_id == 1
        assert sample.features["local_mass"] == 0.7
        assert sample.predicted_zone == "syntax_floor"
        assert sample.actual_zone == "syntax_floor"
        assert sample.confidence == 0.95

    def test_sample_correctness(self):
        """Test sample correctness comparison."""
        correct_sample = ZoneSample(
            fingerprint_id=2,
            features={},
            predicted_zone="semantic_bridge",
            actual_zone="semantic_bridge",
        )
        assert correct_sample.predicted_zone == correct_sample.actual_zone

        incorrect_sample = ZoneSample(
            fingerprint_id=3,
            features={},
            predicted_zone="semantic_bridge",
            actual_zone="long_range",
        )
        assert incorrect_sample.predicted_zone != incorrect_sample.actual_zone


class TestZoneThresholdTuner:
    """Tests for ZoneThresholdTuner class."""

    def test_initialization(self):
        """Test tuner initialization with default thresholds."""
        tuner = ZoneThresholdTuner()

        assert tuner.thresholds is not None
        assert len(tuner._samples) == 0
        assert "syntax_floor" in tuner.thresholds
        assert "semantic_bridge" in tuner.thresholds

    def test_initialization_with_custom_thresholds(self):
        """Test tuner initialization with custom thresholds."""
        custom = {
            "syntax_floor": {"local_mass_min": 0.6},
            "semantic_bridge": {"mid_mass_min": 0.3},
        }
        tuner = ZoneThresholdTuner(initial_thresholds=custom)

        assert tuner.thresholds["syntax_floor"]["local_mass_min"] == 0.6
        assert tuner.thresholds["semantic_bridge"]["mid_mass_min"] == 0.3

    def test_add_sample(self):
        """Test adding samples to tuner."""
        tuner = ZoneThresholdTuner()

        tuner.add_sample(
            fingerprint_id=1,
            features={"local_mass": 0.8, "mid_mass": 0.1, "long_mass": 0.1},
            predicted_zone="syntax_floor",
            actual_zone="syntax_floor",
            confidence=0.9,
        )

        assert len(tuner._samples) == 1
        assert tuner._samples[0].fingerprint_id == 1

    def test_add_multiple_samples(self):
        """Test adding multiple samples."""
        tuner = ZoneThresholdTuner()

        for i in range(10):
            tuner.add_sample(
                fingerprint_id=i,
                features={"local_mass": 0.5 + i * 0.05},
                predicted_zone="syntax_floor",
                actual_zone="syntax_floor" if i % 2 == 0 else "semantic_bridge",
            )

        assert len(tuner._samples) == 10

    def test_add_harness_result(self):
        """Test adding a probe harness result dict."""
        tuner = ZoneThresholdTuner()

        result = {
            "fingerprint_id": 42,
            "features": {"local_mass": 0.8},
            "predicted_zone": "syntax_floor",
            "actual_zone": "syntax_floor",
            "confidence": 0.95,
        }
        tuner.add_harness_result(result)

        assert len(tuner._samples) == 1
        assert tuner._samples[0].fingerprint_id == 42

    def test_compute_accuracy(self):
        """Test accuracy computation returns dict with overall and per-zone."""
        tuner = ZoneThresholdTuner()

        # Add 10 correct and 5 incorrect predictions
        for i in range(15):
            tuner.add_sample(
                fingerprint_id=i,
                features={},
                predicted_zone="syntax_floor",
                actual_zone="syntax_floor" if i < 10 else "semantic_bridge",
            )

        accuracy = tuner.compute_accuracy()

        assert "overall" in accuracy
        assert accuracy["overall"] == pytest.approx(10 / 15, rel=0.01)
        assert "syntax_floor" in accuracy

    def test_compute_accuracy_empty(self):
        """Test accuracy with no samples."""
        tuner = ZoneThresholdTuner()
        accuracy = tuner.compute_accuracy()
        assert accuracy["overall"] == 0.0

    def test_get_confusion_matrix(self):
        """Test confusion matrix generation."""
        tuner = ZoneThresholdTuner()

        # Add some predictions
        tuner.add_sample(1, {}, "syntax_floor", "syntax_floor")
        tuner.add_sample(2, {}, "syntax_floor", "syntax_floor")
        tuner.add_sample(3, {}, "syntax_floor", "semantic_bridge")
        tuner.add_sample(4, {}, "semantic_bridge", "semantic_bridge")

        cm = tuner.get_confusion_matrix()

        assert cm["syntax_floor"]["syntax_floor"] == 2
        assert cm["semantic_bridge"]["syntax_floor"] == 1  # actual=sem_bridge, pred=syn_floor
        assert cm["semantic_bridge"]["semantic_bridge"] == 1

    def test_export_import_thresholds(self):
        """Test exporting and importing thresholds."""
        tuner = ZoneThresholdTuner()
        tuner.thresholds["syntax_floor"]["local_mass_min"] = 0.65

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "thresholds.json")
            exported = tuner.export_thresholds(path)

            assert os.path.exists(path)
            assert exported["thresholds"]["syntax_floor"]["local_mass_min"] == 0.65

            # Create new tuner and import
            tuner2 = ZoneThresholdTuner()
            tuner2.import_thresholds(path)

            assert tuner2.thresholds["syntax_floor"]["local_mass_min"] == 0.65

    def test_save_load_state(self):
        """Test saving and loading tuner state."""
        tuner = ZoneThresholdTuner()

        # Add samples
        tuner.add_sample(1, {"local_mass": 0.8}, "syntax_floor", "syntax_floor")
        tuner.add_sample(2, {"local_mass": 0.5}, "semantic_bridge", "long_range")

        # Modify thresholds
        tuner.thresholds["syntax_floor"]["local_mass_min"] = 0.7

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tuner_state.json")
            tuner.save_state(path)

            assert os.path.exists(path)

            # Load into new tuner
            tuner2 = ZoneThresholdTuner()
            tuner2.load_state(path)

            assert len(tuner2._samples) == 2
            assert tuner2.thresholds["syntax_floor"]["local_mass_min"] == 0.7

    def test_classify_with_thresholds_syntax_floor(self):
        """Test classification of syntax_floor zone."""
        tuner = ZoneThresholdTuner()

        features = {
            "local_mass": 0.8,
            "mid_mass": 0.15,
            "long_mass": 0.05,
            "entropy": 1.5,
        }

        result = tuner._classify_with_thresholds(features, tuner.thresholds)
        assert result == "syntax_floor"

    def test_classify_with_thresholds_long_range(self):
        """Test classification of long_range zone."""
        tuner = ZoneThresholdTuner()

        features = {
            "local_mass": 0.2,
            "mid_mass": 0.3,
            "long_mass": 0.5,
            "entropy": 4.0,
        }

        result = tuner._classify_with_thresholds(features, tuner.thresholds)
        assert result == "long_range"

    def test_get_improvement_suggestions(self):
        """Test improvement suggestions generation."""
        tuner = ZoneThresholdTuner()

        # Add samples with poor accuracy for one zone
        for i in range(20):
            tuner.add_sample(
                fingerprint_id=i,
                features={},
                predicted_zone="syntax_floor",
                actual_zone="long_range",  # All wrong predictions
            )

        suggestions = tuner.get_improvement_suggestions()

        assert len(suggestions) > 0
        # Should suggest improving long_range accuracy
        assert any("long_range" in s for s in suggestions)


class TestFactoryFunction:
    """Tests for create_threshold_tuner factory."""

    def test_create_threshold_tuner_empty(self):
        """Test creating tuner without any paths."""
        tuner = create_threshold_tuner()

        assert tuner is not None
        assert tuner.thresholds is not None

    def test_create_threshold_tuner_with_thresholds_path(self):
        """Test creating tuner with thresholds file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create thresholds file
            thresholds_path = os.path.join(tmpdir, "thresholds.json")
            thresholds = {
                "thresholds": {
                    "syntax_floor": {"local_mass_min": 0.75}
                }
            }
            with open(thresholds_path, "w") as f:
                json.dump(thresholds, f)

            tuner = create_threshold_tuner(thresholds_path=thresholds_path)

            assert tuner.thresholds["syntax_floor"]["local_mass_min"] == 0.75

    def test_create_threshold_tuner_with_state_path(self):
        """Test creating tuner with state file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save initial tuner
            tuner1 = ZoneThresholdTuner()
            tuner1.add_sample(1, {}, "syntax_floor", "syntax_floor")
            state_path = os.path.join(tmpdir, "state.json")
            tuner1.save_state(state_path)

            # Create from state
            tuner2 = create_threshold_tuner(state_path=state_path)

            assert len(tuner2._samples) == 1


class TestZoneConstants:
    """Tests for zone constants."""

    def test_zones_defined(self):
        """Test that all expected zones are defined."""
        expected_zones = [
            "syntax_floor",
            "semantic_bridge",
            "long_range",
            "structure_ripple",
            "diffuse",
        ]

        for zone in expected_zones:
            assert zone in ZONES

    def test_default_thresholds_structure(self):
        """Test default thresholds have expected structure."""
        assert "syntax_floor" in DEFAULT_ZONE_THRESHOLDS
        assert "semantic_bridge" in DEFAULT_ZONE_THRESHOLDS
        assert "long_range" in DEFAULT_ZONE_THRESHOLDS

    def test_default_thresholds_have_mass_params(self):
        """Test default thresholds have mass parameters."""
        # syntax_floor should have local_mass_min
        assert "local_mass_min" in DEFAULT_ZONE_THRESHOLDS["syntax_floor"]

        # long_range should have long_mass_min
        assert "long_mass_min" in DEFAULT_ZONE_THRESHOLDS["long_range"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
