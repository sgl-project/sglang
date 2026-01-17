"""
Checkpoint Manager for Long-Running Discovery Jobs

Provides SQLite-backed state persistence with atomic writes and resume capability.
Enables 8+ hour discovery runs to survive interruptions and resume from any stage.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CheckpointState:
    """
    Serializable state for discovery job resume.

    Captures everything needed to resume a discovery run from any stage.
    """

    run_id: str
    stage: int  # 0-9 (9 stages in pipeline)
    stage_name: str

    # Progress tracking
    fingerprint_ids_processed: List[int] = field(default_factory=list)
    total_fingerprints: int = 0

    # Partial artifact paths (relative to output_dir)
    embeddings_path: Optional[str] = None
    clusters_path: Optional[str] = None
    prototypes_path: Optional[str] = None

    # Zone configuration
    zone_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Timing
    started_at: str = ""
    last_checkpoint_at: str = ""
    elapsed_seconds: float = 0.0

    # Metrics accumulated during run
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Error tracking
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "CheckpointState":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def update_timestamp(self) -> None:
        """Update the last checkpoint timestamp."""
        self.last_checkpoint_at = datetime.now().isoformat()


# Stage names for reference
STAGE_NAMES = [
    "extract",  # 0: Extract fingerprints from DB
    "standardize",  # 1: Standardize features
    "pca",  # 2: PCA dimensionality reduction
    "umap",  # 3: UMAP embedding
    "cluster",  # 4: HDBSCAN clustering
    "zones",  # 5: Zone assignment
    "metadata",  # 6: Cluster metadata
    "prototypes",  # 7: Prototype selection
    "export",  # 8: Export artifacts
    "complete",  # 9: Finalization
]


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================


class CheckpointManager:
    """
    SQLite-backed checkpoint manager for discovery job state.

    Features:
    - Atomic writes (temp file + rename)
    - Stage-by-stage checkpointing
    - Resume from any checkpoint
    - Automatic cleanup on successful completion

    Usage:
        mgr = CheckpointManager(db_path, output_dir)

        # Save checkpoint
        state = CheckpointState(run_id="run-001", stage=3, stage_name="umap")
        mgr.save_checkpoint(state)

        # Resume
        state = mgr.load_checkpoint("run-001")
        if state:
            # Continue from state.stage
    """

    # SQL schema for checkpoints
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS discovery_checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        stage INTEGER NOT NULL,
        stage_name TEXT NOT NULL,
        state_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(run_id, stage)
    );

    CREATE INDEX IF NOT EXISTS idx_checkpoints_run
    ON discovery_checkpoints(run_id);

    CREATE INDEX IF NOT EXISTS idx_checkpoints_created
    ON discovery_checkpoints(created_at);
    """

    def __init__(
        self,
        db_path: str,
        output_dir: str,
        checkpoint_interval_seconds: float = 300.0,  # 5 minutes
    ):
        """
        Initialize checkpoint manager.

        Args:
            db_path: Path to SQLite database (can be same as fingerprints.db)
            output_dir: Directory for discovery outputs (checkpoints stored in subdir)
            checkpoint_interval_seconds: Minimum time between checkpoints
        """
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_interval = checkpoint_interval_seconds

        self._lock = threading.Lock()
        self._last_checkpoint_time: Dict[str, float] = {}

        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_checkpoint(
        self,
        state: CheckpointState,
        force: bool = False,
    ) -> bool:
        """
        Save a checkpoint atomically.

        Args:
            state: Checkpoint state to save
            force: If True, save even if interval hasn't elapsed

        Returns:
            True if checkpoint was saved, False if skipped (too soon)
        """
        import time

        with self._lock:
            # Check if enough time has elapsed
            now = time.time()
            last_time = self._last_checkpoint_time.get(state.run_id, 0)

            if not force and (now - last_time) < self.checkpoint_interval:
                return False

            # Update timestamp
            state.update_timestamp()

            # Save to database atomically
            try:
                with self._get_connection() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO discovery_checkpoints
                        (run_id, stage, stage_name, state_json, created_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                        (
                            state.run_id,
                            state.stage,
                            state.stage_name,
                            state.to_json(),
                        ),
                    )
                    conn.commit()

                self._last_checkpoint_time[state.run_id] = now
                return True

            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
                return False

    def load_checkpoint(self, run_id: str) -> Optional[CheckpointState]:
        """
        Load the most recent checkpoint for a run.

        Args:
            run_id: The run ID to load checkpoint for

        Returns:
            CheckpointState if found, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT state_json FROM discovery_checkpoints
                WHERE run_id = ?
                ORDER BY stage DESC, created_at DESC
                LIMIT 1
            """,
                (run_id,),
            ).fetchone()

            if row:
                return CheckpointState.from_json(row["state_json"])
            return None

    def load_checkpoint_at_stage(
        self,
        run_id: str,
        stage: int,
    ) -> Optional[CheckpointState]:
        """
        Load checkpoint at a specific stage.

        Args:
            run_id: The run ID
            stage: The stage number to load

        Returns:
            CheckpointState if found, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT state_json FROM discovery_checkpoints
                WHERE run_id = ? AND stage = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (run_id, stage),
            ).fetchone()

            if row:
                return CheckpointState.from_json(row["state_json"])
            return None

    def get_resumable_runs(self) -> List[Dict[str, Any]]:
        """
        List all runs that can be resumed.

        Returns:
            List of dicts with run_id, stage, stage_name, last_checkpoint_at
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    run_id,
                    MAX(stage) as stage,
                    stage_name,
                    MAX(created_at) as last_checkpoint
                FROM discovery_checkpoints
                WHERE stage < 9  -- Not complete
                GROUP BY run_id
                ORDER BY last_checkpoint DESC
            """
            ).fetchall()

            return [
                {
                    "run_id": row["run_id"],
                    "stage": row["stage"],
                    "stage_name": row["stage_name"],
                    "last_checkpoint": row["last_checkpoint"],
                }
                for row in rows
            ]

    def clear_checkpoint(self, run_id: str) -> None:
        """
        Clear all checkpoints for a run (call after successful completion).

        Args:
            run_id: The run ID to clear checkpoints for
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                DELETE FROM discovery_checkpoints
                WHERE run_id = ?
            """,
                (run_id,),
            )
            conn.commit()

        # Also clean up partial artifact files
        run_checkpoint_dir = self.checkpoint_dir / run_id
        if run_checkpoint_dir.exists():
            shutil.rmtree(run_checkpoint_dir, ignore_errors=True)

        # Remove from timing cache
        self._last_checkpoint_time.pop(run_id, None)

    def get_checkpoint_history(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get full checkpoint history for a run.

        Args:
            run_id: The run ID

        Returns:
            List of checkpoint records ordered by stage
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT stage, stage_name, created_at, state_json
                FROM discovery_checkpoints
                WHERE run_id = ?
                ORDER BY stage ASC, created_at ASC
            """,
                (run_id,),
            ).fetchall()

            return [
                {
                    "stage": row["stage"],
                    "stage_name": row["stage_name"],
                    "created_at": row["created_at"],
                    "state": CheckpointState.from_json(row["state_json"]),
                }
                for row in rows
            ]

    def save_partial_artifact(
        self,
        run_id: str,
        artifact_name: str,
        data: Any,
        format: str = "parquet",
    ) -> str:
        """
        Save a partial artifact for checkpoint recovery.

        Args:
            run_id: The run ID
            artifact_name: Name of the artifact (e.g., 'embeddings', 'clusters')
            data: Data to save (DataFrame for parquet, dict for json)
            format: 'parquet' or 'json'

        Returns:
            Relative path to saved artifact
        """
        run_dir = self.checkpoint_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "parquet":
            import pandas as pd

            filename = f"{artifact_name}_{timestamp}.parquet"
            filepath = run_dir / filename

            # Atomic write via temp file
            temp_fd, temp_path = tempfile.mkstemp(
                dir=run_dir, prefix=f".{artifact_name}_", suffix=".parquet.tmp"
            )
            try:
                os.close(temp_fd)
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(temp_path, index=False)
                else:
                    pd.DataFrame(data).to_parquet(temp_path, index=False)
                shutil.move(temp_path, filepath)
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        elif format == "json":
            filename = f"{artifact_name}_{timestamp}.json"
            filepath = run_dir / filename

            temp_fd, temp_path = tempfile.mkstemp(
                dir=run_dir, prefix=f".{artifact_name}_", suffix=".json.tmp"
            )
            try:
                with os.fdopen(temp_fd, "w") as f:
                    json.dump(data, f, indent=2)
                shutil.move(temp_path, filepath)
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Return relative path
        return str(filepath.relative_to(self.output_dir))

    def load_partial_artifact(
        self,
        relative_path: str,
        format: str = "parquet",
    ) -> Any:
        """
        Load a partial artifact from checkpoint.

        Args:
            relative_path: Path relative to output_dir
            format: 'parquet' or 'json'

        Returns:
            Loaded data (DataFrame for parquet, dict for json)
        """
        filepath = self.output_dir / relative_path

        if not filepath.exists():
            raise FileNotFoundError(f"Artifact not found: {filepath}")

        if format == "parquet":
            import pandas as pd

            return pd.read_parquet(filepath)
        elif format == "json":
            with open(filepath) as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_checkpoint_state(
    run_id: str,
    stage: int,
    **kwargs,
) -> CheckpointState:
    """
    Create a new checkpoint state.

    Args:
        run_id: Unique run identifier
        stage: Current stage (0-9)
        **kwargs: Additional fields to set

    Returns:
        Initialized CheckpointState
    """
    stage_name = STAGE_NAMES[stage] if stage < len(STAGE_NAMES) else "unknown"

    return CheckpointState(
        run_id=run_id,
        stage=stage,
        stage_name=stage_name,
        started_at=kwargs.pop("started_at", datetime.now().isoformat()),
        **kwargs,
    )


def should_checkpoint(
    current_stage: int,
    last_checkpoint_stage: int,
    elapsed_seconds: float,
    checkpoint_interval: float = 300.0,
) -> bool:
    """
    Determine if we should save a checkpoint.

    Checkpoints at:
    - End of each stage
    - Every checkpoint_interval seconds

    Args:
        current_stage: Current stage number
        last_checkpoint_stage: Stage of last checkpoint
        elapsed_seconds: Seconds since last checkpoint
        checkpoint_interval: Minimum seconds between checkpoints

    Returns:
        True if should checkpoint
    """
    # Always checkpoint on stage change
    if current_stage > last_checkpoint_stage:
        return True

    # Checkpoint by time interval
    if elapsed_seconds >= checkpoint_interval:
        return True

    return False


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import tempfile

    # Test checkpoint manager
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test.db"
        output_dir = f"{tmpdir}/outputs"

        mgr = CheckpointManager(db_path, output_dir)

        # Create and save checkpoint
        state = create_checkpoint_state(
            run_id="test-run-001",
            stage=3,
            total_fingerprints=100000,
            metrics={"fingerprints_processed": 50000},
        )

        saved = mgr.save_checkpoint(state, force=True)
        print(f"Checkpoint saved: {saved}")

        # Load checkpoint
        loaded = mgr.load_checkpoint("test-run-001")
        print(f"Loaded checkpoint: stage={loaded.stage}, name={loaded.stage_name}")

        # List resumable runs
        resumable = mgr.get_resumable_runs()
        print(f"Resumable runs: {resumable}")

        # Save partial artifact
        import pandas as pd

        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        path = mgr.save_partial_artifact("test-run-001", "embeddings", df)
        print(f"Saved artifact: {path}")

        # Load artifact
        loaded_df = mgr.load_partial_artifact(path)
        print(f"Loaded artifact shape: {loaded_df.shape}")

        # Clear checkpoint
        mgr.clear_checkpoint("test-run-001")
        resumable = mgr.get_resumable_runs()
        print(f"After clear, resumable runs: {resumable}")

        print("\nAll tests passed!")
