"""
Unit tests for DiscoveryJobCoordinator and related classes

Tests configuration, stage results, zone classification, and helper methods.
"""

import asyncio
import json
import os
import tempfile
import time
import pytest
import numpy as np
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.coordinator import (
    CoordinatorConfig,
    StageResult,
    DiscoveryResult,
    DiscoveryJobCoordinator,
    run_discovery,
)
from discovery.checkpoint import STAGE_NAMES


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig dataclass."""

    def test_default_config(self):
        """Test config with default values."""
        config = CoordinatorConfig(
            db_path="/path/to/db.sqlite",
            output_dir="/path/to/output",
        )

        assert config.db_path == "/path/to/db.sqlite"
        assert config.output_dir == "/path/to/output"
        assert config.chunk_size == 10000
        assert config.max_memory_gb == 8.0
        assert config.umap_sample_size == 50000
        assert config.checkpoint_interval_seconds == 300.0
        assert config.websocket_port == 9010
        assert config.websocket_enabled is True
        assert config.resume_run_id is None
        assert config.max_runtime_hours is None

    def test_custom_config(self):
        """Test config with custom values."""
        config = CoordinatorConfig(
            db_path="/custom/db.sqlite",
            output_dir="/custom/output",
            chunk_size=5000,
            max_memory_gb=16.0,
            umap_sample_size=100000,
            checkpoint_interval_seconds=600.0,
            websocket_port=9020,
            websocket_enabled=False,
            zone_thresholds_path="/path/to/thresholds.json",
            resume_run_id="run-123",
            max_runtime_hours=8.0,
        )

        assert config.chunk_size == 5000
        assert config.max_memory_gb == 16.0
        assert config.umap_sample_size == 100000
        assert config.checkpoint_interval_seconds == 600.0
        assert config.websocket_port == 9020
        assert config.websocket_enabled is False
        assert config.zone_thresholds_path == "/path/to/thresholds.json"
        assert config.resume_run_id == "run-123"
        assert config.max_runtime_hours == 8.0

    def test_checkpoint_db_path_defaults_to_db_path(self):
        """Test checkpoint_db_path defaults to db_path."""
        config = CoordinatorConfig(
            db_path="/path/to/db.sqlite",
            output_dir="/path/to/output",
        )

        assert config.checkpoint_db_path is None
        # The coordinator will use db_path when checkpoint_db_path is None


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful stage result."""
        result = StageResult(
            stage=3,
            stage_name="umap",
            success=True,
            duration_seconds=125.5,
            metrics={"embeddings_computed": 50000},
        )

        assert result.stage == 3
        assert result.stage_name == "umap"
        assert result.success is True
        assert result.duration_seconds == 125.5
        assert result.metrics["embeddings_computed"] == 50000
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed stage result."""
        result = StageResult(
            stage=4,
            stage_name="cluster",
            success=False,
            duration_seconds=10.5,
            error="Out of memory",
        )

        assert result.success is False
        assert result.error == "Out of memory"

    def test_result_serializable(self):
        """Test stage result is serializable."""
        result = StageResult(
            stage=0,
            stage_name="extract",
            success=True,
            duration_seconds=60.0,
        )

        data = asdict(result)
        json_str = json.dumps(data)
        assert "extract" in json_str


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_successful_discovery_result(self):
        """Test creating a successful discovery result."""
        stage_results = [
            StageResult(stage=i, stage_name=STAGE_NAMES[i], success=True, duration_seconds=10.0)
            for i in range(10)
        ]

        result = DiscoveryResult(
            run_id="run-20260109",
            success=True,
            total_duration_seconds=3600.0,
            stages_completed=10,
            total_fingerprints=100000,
            total_clusters=50,
            zone_distribution={
                "syntax_floor": 40000,
                "semantic_bridge": 35000,
                "long_range": 20000,
                "diffuse": 5000,
            },
            output_paths={
                "export_dir": "/output/discovery_run-20260109",
            },
            stage_results=stage_results,
        )

        assert result.run_id == "run-20260109"
        assert result.success is True
        assert result.stages_completed == 10
        assert result.total_fingerprints == 100000
        assert result.total_clusters == 50
        assert result.zone_distribution["syntax_floor"] == 40000
        assert result.error is None

    def test_failed_discovery_result(self):
        """Test creating a failed discovery result."""
        result = DiscoveryResult(
            run_id="run-failed",
            success=False,
            total_duration_seconds=100.0,
            stages_completed=3,
            total_fingerprints=10000,
            total_clusters=0,
            zone_distribution={},
            output_paths={},
            stage_results=[],
            error="UMAP failed: insufficient memory",
        )

        assert result.success is False
        assert result.stages_completed == 3
        assert result.error == "UMAP failed: insufficient memory"


class TestDiscoveryJobCoordinator:
    """Tests for DiscoveryJobCoordinator class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            output_dir = os.path.join(tmpdir, "outputs")
            os.makedirs(output_dir)
            yield db_path, output_dir

    @pytest.fixture
    def config(self, temp_dirs):
        """Create a test configuration."""
        db_path, output_dir = temp_dirs
        return CoordinatorConfig(
            db_path=db_path,
            output_dir=output_dir,
            websocket_enabled=False,  # Disable for testing
            chunk_size=100,
            umap_sample_size=100,
        )

    def test_initialization(self, config):
        """Test coordinator initialization."""
        coordinator = DiscoveryJobCoordinator(config)

        assert coordinator.config == config
        assert coordinator.output_dir.exists()
        assert coordinator._running is False
        assert coordinator._shutdown_requested is False

    def test_generate_run_id(self, config):
        """Test run ID generation."""
        coordinator = DiscoveryJobCoordinator(config)

        run_id = coordinator._generate_run_id()

        # Should be timestamp-based
        assert len(run_id) == 15  # YYYYMMDD_HHMMSS
        assert "_" in run_id

    def test_classify_zone_syntax_floor(self, config):
        """Test zone classification for syntax_floor."""
        coordinator = DiscoveryJobCoordinator(config)

        # High local mass, low entropy -> syntax_floor
        fingerprint = [0.8, 0.1, 0.1, 1.5]  # local, mid, long, entropy
        zone = coordinator._classify_zone(fingerprint)

        assert zone == "syntax_floor"

    def test_classify_zone_long_range(self, config):
        """Test zone classification for long_range."""
        coordinator = DiscoveryJobCoordinator(config)

        # High long mass, high entropy -> long_range
        fingerprint = [0.2, 0.3, 0.5, 4.0]
        zone = coordinator._classify_zone(fingerprint)

        assert zone == "long_range"

    def test_classify_zone_diffuse(self, config):
        """Test zone classification for diffuse."""
        coordinator = DiscoveryJobCoordinator(config)

        # Very high entropy -> diffuse
        fingerprint = [0.2, 0.3, 0.2, 5.0]
        zone = coordinator._classify_zone(fingerprint)

        assert zone == "diffuse"

    def test_classify_zone_semantic_bridge(self, config):
        """Test zone classification for semantic_bridge (default)."""
        coordinator = DiscoveryJobCoordinator(config)

        # Middle values -> semantic_bridge
        fingerprint = [0.5, 0.3, 0.2, 2.5]
        zone = coordinator._classify_zone(fingerprint)

        assert zone == "semantic_bridge"

    def test_classify_zone_with_custom_thresholds(self, config):
        """Test zone classification with custom thresholds."""
        coordinator = DiscoveryJobCoordinator(config)

        fingerprint = [0.6, 0.2, 0.2, 1.8]

        # With default thresholds, this might be semantic_bridge
        # With custom thresholds, make it syntax_floor
        custom_thresholds = {
            "syntax_floor": {"local_mass_min": 0.5, "entropy_max": 2.5},
        }

        zone = coordinator._classify_zone(fingerprint, custom_thresholds)
        assert zone == "syntax_floor"

    def test_classify_zone_short_fingerprint(self, config):
        """Test zone classification with short fingerprint."""
        coordinator = DiscoveryJobCoordinator(config)

        # Less than 4 elements -> default to semantic_bridge
        fingerprint = [0.5, 0.3]
        zone = coordinator._classify_zone(fingerprint)

        assert zone == "semantic_bridge"

    def test_check_timeout_no_limit(self, config):
        """Test timeout check with no limit."""
        coordinator = DiscoveryJobCoordinator(config)
        coordinator._start_time = time.time()

        # No max_runtime_hours set
        assert coordinator._check_timeout() is False

    def test_check_timeout_not_exceeded(self, config):
        """Test timeout check when not exceeded."""
        config.max_runtime_hours = 1.0
        coordinator = DiscoveryJobCoordinator(config)
        coordinator._start_time = time.time()

        assert coordinator._check_timeout() is False

    def test_check_timeout_exceeded(self, config):
        """Test timeout check when exceeded."""
        config.max_runtime_hours = 0.0001  # Very short
        coordinator = DiscoveryJobCoordinator(config)
        coordinator._start_time = time.time() - 3600  # 1 hour ago

        assert coordinator._check_timeout() is True

    def test_get_zone_distribution_empty(self, config):
        """Test zone distribution with no labels."""
        coordinator = DiscoveryJobCoordinator(config)

        dist = coordinator._get_zone_distribution()
        assert dist == {}

    def test_get_zone_distribution_with_labels(self, config):
        """Test zone distribution with labels."""
        coordinator = DiscoveryJobCoordinator(config)
        coordinator._zone_labels = [
            "syntax_floor", "syntax_floor", "syntax_floor",
            "semantic_bridge", "semantic_bridge",
            "long_range",
        ]

        dist = coordinator._get_zone_distribution()

        assert dist["syntax_floor"] == 3
        assert dist["semantic_bridge"] == 2
        assert dist["long_range"] == 1

    def test_get_stage_metrics_empty(self, config):
        """Test stage metrics with no data."""
        coordinator = DiscoveryJobCoordinator(config)
        coordinator._start_time = time.time()

        metrics = coordinator._get_stage_metrics(0)

        assert "memory_gb" in metrics

    def test_build_result_all_successful(self, config):
        """Test building result when all stages succeed."""
        coordinator = DiscoveryJobCoordinator(config)
        coordinator._run_id = "test-run"
        coordinator._start_time = time.time() - 100
        coordinator._stage_results = [
            StageResult(stage=0, stage_name="extract", success=True, duration_seconds=10.0),
            StageResult(stage=1, stage_name="standardize", success=True, duration_seconds=10.0),
        ]
        coordinator._fingerprints = None
        coordinator._cluster_metadata = None
        coordinator._zone_labels = []

        result = coordinator._build_result()

        assert result.run_id == "test-run"
        assert result.success is True
        assert result.stages_completed == 2

    def test_build_result_with_failure(self, config):
        """Test building result when a stage fails."""
        coordinator = DiscoveryJobCoordinator(config)
        coordinator._run_id = "test-run"
        coordinator._start_time = time.time() - 100
        coordinator._stage_results = [
            StageResult(stage=0, stage_name="extract", success=True, duration_seconds=10.0),
            StageResult(stage=1, stage_name="standardize", success=False, duration_seconds=5.0, error="Memory error"),
        ]
        coordinator._fingerprints = None
        coordinator._cluster_metadata = None
        coordinator._zone_labels = []

        result = coordinator._build_result()

        assert result.success is False
        assert result.stages_completed == 1
        assert result.error == "Memory error"


class TestStageNames:
    """Tests for STAGE_NAMES constant."""

    def test_stage_names_count(self):
        """Test we have 10 stages."""
        assert len(STAGE_NAMES) == 10

    def test_stage_names_order(self):
        """Test stage names are in correct order."""
        expected = [
            "extract", "standardize", "pca", "umap", "cluster",
            "zones", "metadata", "prototypes", "export", "complete",
        ]
        assert STAGE_NAMES == expected


class TestRunDiscoveryFunction:
    """Tests for run_discovery async function."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            output_dir = os.path.join(tmpdir, "outputs")
            os.makedirs(output_dir)
            yield db_path, output_dir

    def test_run_discovery_creates_config(self, temp_dirs):
        """Test run_discovery creates proper config (without actually running)."""
        db_path, output_dir = temp_dirs

        # We can't easily run the full pipeline without a database,
        # but we can verify the function signature works
        # by checking the coordinator is created properly

        with patch('discovery.coordinator.DiscoveryJobCoordinator') as MockCoordinator:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=DiscoveryResult(
                run_id="test",
                success=True,
                total_duration_seconds=0,
                stages_completed=0,
                total_fingerprints=0,
                total_clusters=0,
                zone_distribution={},
                output_paths={},
                stage_results=[],
            ))
            MockCoordinator.return_value = mock_instance

            # Run the async function
            result = asyncio.run(run_discovery(
                db_path=db_path,
                output_dir=output_dir,
                websocket_port=0,  # Disabled
            ))

            # Verify coordinator was created
            MockCoordinator.assert_called_once()
            config = MockCoordinator.call_args[0][0]
            assert config.db_path == db_path
            assert config.output_dir == output_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
