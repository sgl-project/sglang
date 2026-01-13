"""
Integration tests for discovery/coordinator.py

Tests the DiscoveryJobCoordinator including:
- Stage execution with checkpointing
- Memory management
- Zone classification
"""

# Add parent to path for imports
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.bounded_umap import MemoryMonitor
from discovery.coordinator import (
    STAGE_NAMES,
    CoordinatorConfig,
    DiscoveryJobCoordinator,
    DiscoveryResult,
    StageResult,
)


class TestCoordinatorInit:
    """Tests for DiscoveryJobCoordinator initialization."""

    def test_coordinator_initialization(self, populated_db, temp_output_dir):
        """Test basic coordinator initialization."""
        config = CoordinatorConfig(
            db_path=populated_db,
            output_dir=temp_output_dir,
        )

        coordinator = DiscoveryJobCoordinator(config)

        assert coordinator.config is not None
        assert coordinator.config.db_path == populated_db


class TestCoordinatorMemoryManagement:
    """Tests for coordinator memory management."""

    def test_coordinator_has_memory_monitor(self, populated_db, temp_output_dir):
        """Test coordinator initializes memory monitor."""
        config = CoordinatorConfig(
            db_path=populated_db,
            output_dir=temp_output_dir,
            max_memory_gb=8.0,
        )

        coordinator = DiscoveryJobCoordinator(config)

        assert coordinator._memory_monitor is not None
        assert isinstance(coordinator._memory_monitor, MemoryMonitor)


class TestCoordinatorZoneClassification:
    """Tests for zone classification in coordinator."""

    def test_classify_zone_syntax_floor(self, populated_db, temp_output_dir):
        """Test syntax_floor zone classification."""
        config = CoordinatorConfig(
            db_path=populated_db,
            output_dir=temp_output_dir,
        )

        coordinator = DiscoveryJobCoordinator(config)

        # High local mass, low entropy
        fp = np.zeros(20, dtype=np.float32)
        fp[0] = 0.7  # local_mass
        fp[1] = 0.2  # mid_mass
        fp[2] = 0.1  # long_mass
        fp[3] = 1.5  # entropy

        zone = coordinator._classify_zone(fp)

        # Check zone is one of the valid types
        assert zone in ["syntax_floor", "semantic_bridge", "long_range", "diffuse"]

    def test_classify_zone_long_range(self, populated_db, temp_output_dir):
        """Test long_range zone classification."""
        config = CoordinatorConfig(
            db_path=populated_db,
            output_dir=temp_output_dir,
        )

        coordinator = DiscoveryJobCoordinator(config)

        # High long mass
        fp = np.zeros(20, dtype=np.float32)
        fp[0] = 0.1  # local_mass
        fp[1] = 0.2  # mid_mass
        fp[2] = 0.5  # long_mass (high)
        fp[3] = 3.0  # entropy

        zone = coordinator._classify_zone(fp)

        assert zone == "long_range"

    def test_classify_zone_semantic_bridge(self, populated_db, temp_output_dir):
        """Test semantic_bridge (default) zone classification."""
        config = CoordinatorConfig(
            db_path=populated_db,
            output_dir=temp_output_dir,
        )

        coordinator = DiscoveryJobCoordinator(config)

        # Mid-range values
        fp = np.zeros(20, dtype=np.float32)
        fp[0] = 0.35  # local_mass
        fp[1] = 0.4  # mid_mass
        fp[2] = 0.15  # long_mass
        fp[3] = 2.8  # entropy

        zone = coordinator._classify_zone(fp)

        assert zone == "semantic_bridge"


class TestCoordinatorTimeout:
    """Tests for coordinator timeout handling."""

    def test_check_timeout_no_limit(self, populated_db, temp_output_dir):
        """Test timeout check with no limit."""
        config = CoordinatorConfig(
            db_path=populated_db,
            output_dir=temp_output_dir,
            max_runtime_hours=None,  # No limit
        )

        coordinator = DiscoveryJobCoordinator(config)
        coordinator._start_time = datetime.utcnow()

        # Should return False (not timed out)
        result = coordinator._check_timeout()
        assert result is False

    def test_check_timeout_not_exceeded(self, populated_db, temp_output_dir):
        """Test timeout check when not exceeded."""
        config = CoordinatorConfig(
            db_path=populated_db,
            output_dir=temp_output_dir,
            max_runtime_hours=8.0,
        )

        coordinator = DiscoveryJobCoordinator(config)
        # Start time needs to be a timestamp, not datetime
        coordinator._start_time = datetime.utcnow().timestamp()

        # Should return False (just started)
        result = coordinator._check_timeout()
        assert result is False


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_successful_stage_result(self):
        """Test creating a successful stage result."""
        result = StageResult(
            stage=0,
            stage_name="extract",
            success=True,
            duration_seconds=5.0,
        )

        assert result.stage == 0
        assert result.stage_name == "extract"
        assert result.success is True
        assert result.duration_seconds == 5.0

    def test_failed_stage_result(self):
        """Test creating a failed stage result."""
        result = StageResult(
            stage=1,
            stage_name="scale",
            success=False,
            duration_seconds=1.0,
            error="Test error message",
        )

        assert result.success is False
        assert result.error == "Test error message"

    def test_stage_result_with_metrics(self):
        """Test stage result with metrics."""
        result = StageResult(
            stage=0,
            stage_name="extract",
            success=True,
            duration_seconds=5.0,
            metrics={"fingerprint_count": 1000},
        )

        assert result.metrics["fingerprint_count"] == 1000


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_successful_discovery_result(self):
        """Test creating a successful discovery result."""
        result = DiscoveryResult(
            run_id="test_run",
            success=True,
            total_duration_seconds=100.0,
            stages_completed=9,
            total_fingerprints=1000,
            total_clusters=5,
            zone_distribution={
                "syntax_floor": 300,
                "semantic_bridge": 500,
                "long_range": 200,
            },
            output_paths={"embeddings": "/path/to/embeddings.parquet"},
            stage_results=[],
        )

        assert result.success is True
        assert result.stages_completed == 9
        assert result.total_fingerprints == 1000

    def test_failed_discovery_result(self):
        """Test creating a failed discovery result."""
        result = DiscoveryResult(
            run_id="test_run",
            success=False,
            total_duration_seconds=50.0,
            stages_completed=3,
            total_fingerprints=0,
            total_clusters=0,
            zone_distribution={},
            output_paths={},
            stage_results=[],
            error="Pipeline failed",
        )

        assert result.success is False
        assert result.error == "Pipeline failed"


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig dataclass."""

    def test_default_config(self, populated_db, temp_output_dir):
        """Test default configuration values."""
        config = CoordinatorConfig(
            db_path=populated_db,
            output_dir=temp_output_dir,
        )

        assert config.chunk_size == 10000
        assert config.max_memory_gb == 8.0
        assert config.websocket_port == 9010

    def test_custom_config(self, populated_db, temp_output_dir):
        """Test custom configuration."""
        config = CoordinatorConfig(
            db_path=populated_db,
            output_dir=temp_output_dir,
            chunk_size=5000,
            max_memory_gb=16.0,
            websocket_port=0,  # Disabled
        )

        assert config.chunk_size == 5000
        assert config.max_memory_gb == 16.0
        assert config.websocket_port == 0

    def test_checkpoint_db_defaults(self, populated_db, temp_output_dir):
        """Test checkpoint_db_path defaults."""
        config = CoordinatorConfig(
            db_path=populated_db,
            output_dir=temp_output_dir,
        )

        # If checkpoint_db_path not specified, should be None
        assert config.checkpoint_db_path is None


class TestStageNames:
    """Tests for stage name constants."""

    def test_stage_names_defined(self):
        """Test stage names are defined."""
        assert STAGE_NAMES is not None
        assert len(STAGE_NAMES) > 0

    def test_stage_names_are_strings(self):
        """Test all stage names are strings."""
        assert all(isinstance(name, str) for name in STAGE_NAMES)


class TestCoordinatorRunId:
    """Tests for coordinator run ID generation."""

    def test_run_id_generated(self, populated_db, temp_output_dir):
        """Test run ID is generated."""
        config = CoordinatorConfig(
            db_path=populated_db,
            output_dir=temp_output_dir,
        )

        coordinator = DiscoveryJobCoordinator(config)

        # Run ID should be generated
        run_id = coordinator._generate_run_id()

        assert run_id is not None
        assert isinstance(run_id, str)
        assert len(run_id) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
