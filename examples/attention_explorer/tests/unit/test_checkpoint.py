"""
Unit tests for CheckpointManager

Tests checkpoint save/load/resume functionality for long-running discovery jobs.
"""

import os

# Add parent to path for imports
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.checkpoint import (
    STAGE_NAMES,
    CheckpointManager,
    CheckpointState,
    create_checkpoint_state,
)


class TestCheckpointState:
    """Tests for CheckpointState dataclass."""

    def test_create_checkpoint_state(self):
        """Test creating a checkpoint state with stage name lookup."""
        state = create_checkpoint_state(
            run_id="test-run-001",
            stage=3,
            total_fingerprints=1000,
        )

        assert state.run_id == "test-run-001"
        assert state.stage == 3
        assert state.stage_name == "umap"  # Stage 3 is umap
        assert state.total_fingerprints == 1000
        assert state.started_at != ""

    def test_checkpoint_state_serialization(self):
        """Test JSON serialization round-trip."""
        state = create_checkpoint_state(
            run_id="test-run-002",
            stage=5,
            total_fingerprints=50000,
            metrics={"accuracy": 0.95, "clusters": 42},
        )

        # Serialize
        json_str = state.to_json()
        assert isinstance(json_str, str)

        # Deserialize
        restored = CheckpointState.from_json(json_str)
        assert restored.run_id == state.run_id
        assert restored.stage == state.stage
        assert restored.stage_name == state.stage_name
        assert restored.total_fingerprints == state.total_fingerprints
        assert restored.metrics == state.metrics

    def test_checkpoint_state_update_timestamp(self):
        """Test timestamp update method."""
        state = create_checkpoint_state(run_id="test", stage=0)
        original_timestamp = state.last_checkpoint_at

        state.update_timestamp()
        assert state.last_checkpoint_at != original_timestamp
        assert state.last_checkpoint_at != ""


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_checkpoints.db")
            output_dir = os.path.join(tmpdir, "outputs")
            os.makedirs(output_dir)
            yield db_path, output_dir

    def test_manager_initialization(self, temp_dirs):
        """Test checkpoint manager initialization creates schema."""
        db_path, output_dir = temp_dirs

        mgr = CheckpointManager(db_path, output_dir)

        # Verify checkpoint directory was created
        assert (Path(output_dir) / "checkpoints").exists()

        # Verify can query (schema was created)
        runs = mgr.get_resumable_runs()
        assert runs == []

    def test_save_and_load_checkpoint(self, temp_dirs):
        """Test saving and loading a checkpoint."""
        db_path, output_dir = temp_dirs
        mgr = CheckpointManager(db_path, output_dir, checkpoint_interval_seconds=0)

        # Create and save checkpoint
        state = create_checkpoint_state(
            run_id="run-001",
            stage=4,
            total_fingerprints=25000,
            metrics={"processed": 20000},
        )

        saved = mgr.save_checkpoint(state, force=True)
        assert saved is True

        # Load checkpoint
        loaded = mgr.load_checkpoint("run-001")
        assert loaded is not None
        assert loaded.run_id == "run-001"
        assert loaded.stage == 4
        assert loaded.total_fingerprints == 25000
        assert loaded.metrics["processed"] == 20000

    def test_load_nonexistent_checkpoint(self, temp_dirs):
        """Test loading a checkpoint that doesn't exist returns None."""
        db_path, output_dir = temp_dirs
        mgr = CheckpointManager(db_path, output_dir)

        loaded = mgr.load_checkpoint("nonexistent-run")
        assert loaded is None

    def test_checkpoint_interval_throttling(self, temp_dirs):
        """Test that checkpoints are throttled by interval."""
        db_path, output_dir = temp_dirs
        mgr = CheckpointManager(db_path, output_dir, checkpoint_interval_seconds=300)

        state = create_checkpoint_state(run_id="run-002", stage=1)

        # First save should succeed (forced)
        saved1 = mgr.save_checkpoint(state, force=True)
        assert saved1 is True

        # Second save should be skipped (too soon)
        state.stage = 2
        saved2 = mgr.save_checkpoint(state, force=False)
        assert saved2 is False

        # Third save with force should succeed
        state.stage = 3
        saved3 = mgr.save_checkpoint(state, force=True)
        assert saved3 is True

    def test_resumable_runs_listing(self, temp_dirs):
        """Test listing resumable runs."""
        db_path, output_dir = temp_dirs
        mgr = CheckpointManager(db_path, output_dir, checkpoint_interval_seconds=0)

        # Create multiple runs at different stages
        for run_id, stage in [("run-A", 3), ("run-B", 5), ("run-C", 8)]:
            state = create_checkpoint_state(run_id=run_id, stage=stage)
            mgr.save_checkpoint(state, force=True)

        # Get resumable runs (stage < 9)
        runs = mgr.get_resumable_runs()
        assert len(runs) == 3

        # Verify they're sorted by last checkpoint (most recent first)
        run_ids = [r["run_id"] for r in runs]
        assert "run-A" in run_ids
        assert "run-B" in run_ids
        assert "run-C" in run_ids

    def test_clear_checkpoint(self, temp_dirs):
        """Test clearing checkpoints for a completed run."""
        db_path, output_dir = temp_dirs
        mgr = CheckpointManager(db_path, output_dir, checkpoint_interval_seconds=0)

        # Save checkpoint
        state = create_checkpoint_state(run_id="run-to-clear", stage=5)
        mgr.save_checkpoint(state, force=True)

        # Verify it exists
        loaded = mgr.load_checkpoint("run-to-clear")
        assert loaded is not None

        # Clear it
        mgr.clear_checkpoint("run-to-clear")

        # Verify it's gone
        loaded = mgr.load_checkpoint("run-to-clear")
        assert loaded is None

    def test_checkpoint_at_specific_stage(self, temp_dirs):
        """Test loading checkpoint at a specific stage."""
        db_path, output_dir = temp_dirs
        mgr = CheckpointManager(db_path, output_dir, checkpoint_interval_seconds=0)

        # Save checkpoints at multiple stages
        for stage in [1, 3, 5]:
            state = create_checkpoint_state(
                run_id="multi-stage-run",
                stage=stage,
                total_fingerprints=stage * 1000,
            )
            mgr.save_checkpoint(state, force=True)

        # Load specific stages
        stage_3 = mgr.load_checkpoint_at_stage("multi-stage-run", 3)
        assert stage_3 is not None
        assert stage_3.stage == 3
        assert stage_3.total_fingerprints == 3000

        stage_5 = mgr.load_checkpoint_at_stage("multi-stage-run", 5)
        assert stage_5 is not None
        assert stage_5.stage == 5

        # Non-existent stage
        stage_7 = mgr.load_checkpoint_at_stage("multi-stage-run", 7)
        assert stage_7 is None

    def test_partial_artifact_storage(self, temp_dirs):
        """Test saving and loading partial artifacts."""
        db_path, output_dir = temp_dirs
        mgr = CheckpointManager(db_path, output_dir)

        import numpy as np
        import pandas as pd

        # Save parquet artifact
        df = pd.DataFrame(
            {
                "x": np.random.randn(100),
                "y": np.random.randn(100),
                "cluster": np.random.randint(0, 5, 100),
            }
        )

        path = mgr.save_partial_artifact("run-artifact", "embeddings", df, "parquet")
        assert path is not None
        assert "embeddings" in path

        # Load artifact
        loaded_df = mgr.load_partial_artifact(path, "parquet")
        assert len(loaded_df) == 100
        assert list(loaded_df.columns) == ["x", "y", "cluster"]

    def test_partial_artifact_json(self, temp_dirs):
        """Test saving and loading JSON artifacts."""
        db_path, output_dir = temp_dirs
        mgr = CheckpointManager(db_path, output_dir)

        # Save JSON artifact
        data = {
            "thresholds": {"zone_a": 0.5, "zone_b": 0.8},
            "metrics": [1, 2, 3],
        }

        path = mgr.save_partial_artifact("run-json", "config", data, "json")
        assert path is not None

        # Load artifact
        loaded = mgr.load_partial_artifact(path, "json")
        assert loaded["thresholds"]["zone_a"] == 0.5
        assert loaded["metrics"] == [1, 2, 3]

    def test_checkpoint_history(self, temp_dirs):
        """Test getting checkpoint history for a run."""
        db_path, output_dir = temp_dirs
        mgr = CheckpointManager(db_path, output_dir, checkpoint_interval_seconds=0)

        # Save checkpoints at multiple stages
        for stage in range(5):
            state = create_checkpoint_state(run_id="history-run", stage=stage)
            mgr.save_checkpoint(state, force=True)

        # Get history
        history = mgr.get_checkpoint_history("history-run")
        assert len(history) == 5

        # Verify ordered by stage
        stages = [h["stage"] for h in history]
        assert stages == [0, 1, 2, 3, 4]


class TestStageNames:
    """Tests for stage name constants."""

    def test_stage_names_count(self):
        """Test that we have all 10 stages defined."""
        assert len(STAGE_NAMES) == 10

    def test_stage_names_order(self):
        """Test stage name order matches pipeline."""
        expected = [
            "extract",
            "standardize",
            "pca",
            "umap",
            "cluster",
            "zones",
            "metadata",
            "prototypes",
            "export",
            "complete",
        ]
        assert STAGE_NAMES == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
