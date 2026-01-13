"""
Tests for Checkpoint Manager

Tests the SQLite-backed checkpoint persistence and resume capability.
"""

import json
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from discovery.checkpoint import (
    STAGE_NAMES,
    CheckpointManager,
    CheckpointState,
    create_checkpoint_state,
    should_checkpoint,
)


class TestCheckpointState:
    """Tests for CheckpointState dataclass."""

    def test_create_basic_state(self):
        """Test basic checkpoint state creation."""
        state = CheckpointState(
            run_id="test-run-001",
            stage=3,
            stage_name="umap",
        )

        assert state.run_id == "test-run-001"
        assert state.stage == 3
        assert state.stage_name == "umap"
        assert state.total_fingerprints == 0
        assert state.metrics == {}

    def test_state_to_json(self):
        """Test serialization to JSON."""
        state = CheckpointState(
            run_id="test-run",
            stage=2,
            stage_name="pca",
            total_fingerprints=50000,
            metrics={"variance_explained": 0.95},
        )

        json_str = state.to_json()
        data = json.loads(json_str)

        assert data["run_id"] == "test-run"
        assert data["stage"] == 2
        assert data["total_fingerprints"] == 50000
        assert data["metrics"]["variance_explained"] == 0.95

    def test_state_from_json(self):
        """Test deserialization from JSON."""
        json_str = json.dumps(
            {
                "run_id": "test-run",
                "stage": 4,
                "stage_name": "cluster",
                "fingerprint_ids_processed": [1, 2, 3],
                "total_fingerprints": 1000,
                "embeddings_path": None,
                "clusters_path": None,
                "prototypes_path": None,
                "zone_thresholds": {"syntax_floor": {"local_mass_min": 0.7}},
                "started_at": "2026-01-10T10:00:00",
                "last_checkpoint_at": "2026-01-10T10:30:00",
                "elapsed_seconds": 1800.0,
                "metrics": {"clusters_found": 42},
                "errors": [],
            }
        )

        state = CheckpointState.from_json(json_str)

        assert state.run_id == "test-run"
        assert state.stage == 4
        assert state.total_fingerprints == 1000
        assert state.zone_thresholds["syntax_floor"]["local_mass_min"] == 0.7

    def test_update_timestamp(self):
        """Test timestamp update."""
        state = CheckpointState(
            run_id="test",
            stage=0,
            stage_name="extract",
        )

        assert state.last_checkpoint_at == ""
        state.update_timestamp()
        assert state.last_checkpoint_at != ""
        # Should be a valid ISO timestamp
        datetime.fromisoformat(state.last_checkpoint_at)


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        Path(path).unlink(missing_ok=True)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_db, temp_dir):
        """Create checkpoint manager."""
        return CheckpointManager(temp_db, temp_dir)

    def test_init_creates_schema(self, temp_db, temp_dir):
        """Test that initialization creates database schema."""
        manager = CheckpointManager(temp_db, temp_dir)

        # Check tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "discovery_checkpoints" in tables

    def test_save_and_load_checkpoint(self, manager):
        """Test saving and loading a checkpoint."""
        state = create_checkpoint_state(
            run_id="test-run-001",
            stage=3,
            total_fingerprints=10000,
            metrics={"embeddings_shape": [10000, 2]},
        )

        # Save
        saved = manager.save_checkpoint(state, force=True)
        assert saved is True

        # Load
        loaded = manager.load_checkpoint("test-run-001")

        assert loaded is not None
        assert loaded.run_id == "test-run-001"
        assert loaded.stage == 3
        assert loaded.total_fingerprints == 10000

    def test_load_latest_checkpoint(self, manager):
        """Test that loading returns the most recent checkpoint."""
        # Save multiple checkpoints for same run
        for stage in range(5):
            state = create_checkpoint_state(
                run_id="test-run",
                stage=stage,
                total_fingerprints=stage * 1000,
            )
            manager.save_checkpoint(state, force=True)

        # Load should return the latest (stage 4)
        loaded = manager.load_checkpoint("test-run")

        assert loaded.stage == 4
        assert loaded.total_fingerprints == 4000

    def test_load_checkpoint_at_stage(self, manager):
        """Test loading checkpoint at specific stage."""
        # Save checkpoints
        for stage in range(5):
            state = create_checkpoint_state(
                run_id="test-run",
                stage=stage,
                total_fingerprints=stage * 1000,
            )
            manager.save_checkpoint(state, force=True)

        # Load specific stage
        loaded = manager.load_checkpoint_at_stage("test-run", 2)

        assert loaded is not None
        assert loaded.stage == 2
        assert loaded.total_fingerprints == 2000

    def test_load_nonexistent_checkpoint(self, manager):
        """Test loading non-existent checkpoint returns None."""
        loaded = manager.load_checkpoint("nonexistent-run")
        assert loaded is None

    def test_get_resumable_runs(self, manager):
        """Test listing resumable runs."""
        # Create incomplete runs
        for i in range(3):
            state = create_checkpoint_state(
                run_id=f"run-{i}",
                stage=i + 1,  # Stages 1, 2, 3 (not complete)
            )
            manager.save_checkpoint(state, force=True)

        # Create complete run
        complete_state = create_checkpoint_state(
            run_id="complete-run",
            stage=9,  # Complete
        )
        manager.save_checkpoint(complete_state, force=True)

        resumable = manager.get_resumable_runs()

        # Should have 3 resumable runs (not the complete one)
        assert len(resumable) == 3
        run_ids = [r["run_id"] for r in resumable]
        assert "complete-run" not in run_ids

    def test_clear_checkpoint(self, manager, temp_dir):
        """Test clearing checkpoint."""
        state = create_checkpoint_state(
            run_id="test-run",
            stage=3,
        )
        manager.save_checkpoint(state, force=True)

        # Verify it exists
        assert manager.load_checkpoint("test-run") is not None

        # Clear
        manager.clear_checkpoint("test-run")

        # Verify it's gone
        assert manager.load_checkpoint("test-run") is None

    def test_checkpoint_interval(self, manager):
        """Test checkpoint interval throttling."""
        state = create_checkpoint_state(
            run_id="test-run",
            stage=1,
        )

        # First save should succeed
        assert manager.save_checkpoint(state, force=False) is True

        # Immediate second save should be skipped (too soon)
        state.stage = 2
        assert manager.save_checkpoint(state, force=False) is False

        # Force save should succeed
        assert manager.save_checkpoint(state, force=True) is True

    def test_get_checkpoint_history(self, manager):
        """Test checkpoint history retrieval."""
        # Save multiple checkpoints
        for stage in range(4):
            state = create_checkpoint_state(
                run_id="test-run",
                stage=stage,
            )
            manager.save_checkpoint(state, force=True)

        history = manager.get_checkpoint_history("test-run")

        assert len(history) == 4
        assert history[0]["stage"] == 0
        assert history[3]["stage"] == 3

    def test_save_partial_artifact_parquet(self, manager):
        """Test saving parquet artifact."""
        df = pd.DataFrame(
            {
                "x": np.random.randn(100),
                "y": np.random.randn(100),
                "cluster": np.random.randint(0, 5, 100),
            }
        )

        path = manager.save_partial_artifact(
            "test-run",
            "embeddings",
            df,
            format="parquet",
        )

        assert path is not None
        assert path.endswith(".parquet")

        # Load and verify
        loaded = manager.load_partial_artifact(path, format="parquet")
        assert len(loaded) == 100
        assert list(loaded.columns) == ["x", "y", "cluster"]

    def test_save_partial_artifact_json(self, manager):
        """Test saving JSON artifact."""
        data = {
            "clusters": [
                {"id": 0, "size": 100, "zone": "syntax_floor"},
                {"id": 1, "size": 200, "zone": "semantic_bridge"},
            ],
            "metadata": {"version": 1},
        }

        path = manager.save_partial_artifact(
            "test-run",
            "clusters",
            data,
            format="json",
        )

        assert path.endswith(".json")

        loaded = manager.load_partial_artifact(path, format="json")
        assert len(loaded["clusters"]) == 2
        assert loaded["metadata"]["version"] == 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_checkpoint_state(self):
        """Test checkpoint state creation helper."""
        state = create_checkpoint_state(
            run_id="run-123",
            stage=5,
            total_fingerprints=75000,
            metrics={"key": "value"},
        )

        assert state.run_id == "run-123"
        assert state.stage == 5
        assert state.stage_name == STAGE_NAMES[5]
        assert state.total_fingerprints == 75000
        assert state.metrics == {"key": "value"}

    def test_should_checkpoint_on_stage_change(self):
        """Test checkpointing on stage change."""
        assert (
            should_checkpoint(
                current_stage=3,
                last_checkpoint_stage=2,
                elapsed_seconds=10,
                checkpoint_interval=300,
            )
            is True
        )

    def test_should_checkpoint_on_interval(self):
        """Test checkpointing on time interval."""
        assert (
            should_checkpoint(
                current_stage=3,
                last_checkpoint_stage=3,
                elapsed_seconds=400,  # > 300s interval
                checkpoint_interval=300,
            )
            is True
        )

    def test_should_not_checkpoint_too_soon(self):
        """Test skipping checkpoint when too soon."""
        assert (
            should_checkpoint(
                current_stage=3,
                last_checkpoint_stage=3,
                elapsed_seconds=100,  # < 300s interval
                checkpoint_interval=300,
            )
            is False
        )


class TestResumeScenarios:
    """Tests for resume scenarios."""

    @pytest.fixture
    def manager(self):
        """Create manager with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "checkpoints.db")
            yield CheckpointManager(db_path, tmpdir)

    def test_resume_from_extract_stage(self, manager):
        """Test resume from extract stage."""
        # Simulate interrupted run at stage 1
        state = create_checkpoint_state(
            run_id="interrupted-run",
            stage=1,
            total_fingerprints=50000,
        )
        state.fingerprint_ids_processed = list(range(50000))
        manager.save_checkpoint(state, force=True)

        # Resume
        loaded = manager.load_checkpoint("interrupted-run")

        assert loaded is not None
        assert loaded.stage == 1
        assert len(loaded.fingerprint_ids_processed) == 50000

    def test_resume_from_umap_stage(self, manager):
        """Test resume from UMAP stage with partial embeddings."""
        # Save partial embeddings
        partial_embeddings = pd.DataFrame(
            {
                "id": range(25000),
                "x": np.random.randn(25000),
                "y": np.random.randn(25000),
            }
        )
        path = manager.save_partial_artifact(
            "run-umap",
            "embeddings_partial",
            partial_embeddings,
        )

        # Save checkpoint
        state = create_checkpoint_state(
            run_id="run-umap",
            stage=3,
            total_fingerprints=100000,
        )
        state.embeddings_path = path
        state.metrics = {"embeddings_computed": 25000}
        manager.save_checkpoint(state, force=True)

        # Resume
        loaded = manager.load_checkpoint("run-umap")
        assert loaded.stage == 3
        assert loaded.embeddings_path is not None

        # Load partial embeddings
        embeddings = manager.load_partial_artifact(loaded.embeddings_path)
        assert len(embeddings) == 25000

    def test_resume_preserves_zone_thresholds(self, manager):
        """Test that resume preserves custom zone thresholds."""
        custom_thresholds = {
            "syntax_floor": {"local_mass_min": 0.8, "entropy_max": 1.5},
            "long_range": {"long_mass_min": 0.4},
        }

        state = create_checkpoint_state(
            run_id="run-zones",
            stage=5,
        )
        state.zone_thresholds = custom_thresholds
        manager.save_checkpoint(state, force=True)

        loaded = manager.load_checkpoint("run-zones")
        assert loaded.zone_thresholds == custom_thresholds

    def test_multiple_runs_independent(self, manager):
        """Test that multiple runs have independent checkpoints."""
        # Create two runs
        for run_id in ["run-a", "run-b"]:
            state = create_checkpoint_state(
                run_id=run_id,
                stage=3 if run_id == "run-a" else 5,
            )
            manager.save_checkpoint(state, force=True)

        # Verify independence
        run_a = manager.load_checkpoint("run-a")
        run_b = manager.load_checkpoint("run-b")

        assert run_a.stage == 3
        assert run_b.stage == 5

        # Clear one doesn't affect other
        manager.clear_checkpoint("run-a")
        assert manager.load_checkpoint("run-a") is None
        assert manager.load_checkpoint("run-b") is not None


class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    @pytest.fixture
    def manager(self):
        """Create manager with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "checkpoints.db")
            yield CheckpointManager(db_path, tmpdir)

    def test_checkpoint_with_error(self, manager):
        """Test checkpoint includes error information."""
        state = create_checkpoint_state(
            run_id="failed-run",
            stage=4,
        )
        state.errors = [{"stage": 4, "error": "HDBSCAN failed: not enough memory"}]
        manager.save_checkpoint(state, force=True)

        loaded = manager.load_checkpoint("failed-run")
        assert len(loaded.errors) == 1
        assert "memory" in loaded.errors[0]["error"]

    def test_atomic_write_integrity(self, manager):
        """Test that checkpoint writes are atomic."""
        state = create_checkpoint_state(
            run_id="atomic-test",
            stage=3,
            total_fingerprints=100000,
        )

        # Save multiple times rapidly
        for i in range(10):
            state.metrics = {"iteration": i}
            manager.save_checkpoint(state, force=True)

        # Final state should be consistent
        loaded = manager.load_checkpoint("atomic-test")
        assert loaded.metrics["iteration"] == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
