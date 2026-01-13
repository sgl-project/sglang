"""
Unit tests for RAPIDS Sidecar utility classes and functions

Tests FingerprintStorage, ClusteringBackend, and helper utilities.
"""

import os
import sqlite3

# Add parent to path for imports
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rapids_sidecar import (
    HAS_RAPIDS,
    HAS_SKLEARN,
    HAS_ZMQ,
    SIDECAR_VERSION,
    ClusteringBackend,
    FingerprintStorage,
)


class TestClusteringBackend:
    """Tests for ClusteringBackend enum."""

    def test_backend_values(self):
        """Test backend enum values."""
        assert ClusteringBackend.RAPIDS.value == "rapids"
        assert ClusteringBackend.CPU.value == "cpu"
        assert ClusteringBackend.ONLINE.value == "online"
        assert ClusteringBackend.NONE.value == "none"

    def test_backend_comparison(self):
        """Test backend enum comparison."""
        assert ClusteringBackend.RAPIDS != ClusteringBackend.CPU
        assert ClusteringBackend.RAPIDS == ClusteringBackend.RAPIDS


class TestSidecarVersion:
    """Tests for sidecar version constant."""

    def test_version_format(self):
        """Test version string format."""
        assert isinstance(SIDECAR_VERSION, str)
        parts = SIDECAR_VERSION.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)


class TestFingerprintStorage:
    """Tests for FingerprintStorage class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database with full schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_fingerprints.db")

            # Create schema matching discovery/schema.sql
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    name TEXT,
                    model_id TEXT,
                    request_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    session_id TEXT,
                    step INTEGER NOT NULL,
                    token_id INTEGER,
                    token_text TEXT,
                    think_phase TEXT,
                    segment_idx INTEGER DEFAULT 0,
                    fingerprint BLOB NOT NULL,
                    manifold_zone TEXT,
                    manifold_confidence REAL,
                    cluster_id INTEGER DEFAULT -1,
                    cluster_probability REAL,
                    top_expert_ids BLOB,
                    router_entropy REAL,
                    expert_load_balance REAL,
                    top_k_positions BLOB,
                    top_k_scores BLOB,
                    sink_token_mass REAL,
                    capture_layer_ids BLOB,
                    schema_version INTEGER DEFAULT 1,
                    model_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(request_id, step)
                );

                CREATE INDEX IF NOT EXISTS idx_fingerprints_session
                    ON fingerprints(session_id);
                CREATE INDEX IF NOT EXISTS idx_fingerprints_request
                    ON fingerprints(request_id);
            """
            )
            conn.commit()
            conn.close()

            yield db_path

    def test_initialization(self, temp_db):
        """Test storage initialization."""
        storage = FingerprintStorage(db_path=temp_db, session_id="test-session")

        assert storage.db_path == temp_db
        assert storage.session_id == "test-session"
        assert storage._initialized is False

    def test_default_session_id(self, temp_db):
        """Test default session ID generation."""
        storage = FingerprintStorage(db_path=temp_db)

        assert storage.session_id.startswith("sidecar_")

    def test_ensure_initialized(self, temp_db):
        """Test database initialization."""
        storage = FingerprintStorage(db_path=temp_db, session_id="test-session")
        storage._ensure_initialized()

        assert storage._initialized is True
        assert storage._conn is not None

        # Verify session was created
        cursor = storage._conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", ("test-session",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "test-session"

    def test_store_fingerprint(self, temp_db):
        """Test storing a fingerprint."""
        storage = FingerprintStorage(db_path=temp_db, session_id="test-session")
        storage._ensure_initialized()

        fingerprint = np.random.randn(20).astype(np.float32)
        storage.store(
            request_id="req-001",
            step=0,
            fingerprint=fingerprint,
            metadata={"prompt_type": "code"},
        )

        # Force flush
        storage.flush()

        # Verify stored
        cursor = storage._conn.execute(
            "SELECT request_id, step FROM fingerprints WHERE request_id = ?",
            ("req-001",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "req-001"
        assert row[1] == 0

    def test_store_multiple_fingerprints(self, temp_db):
        """Test storing multiple fingerprints."""
        storage = FingerprintStorage(db_path=temp_db, session_id="test-session")
        storage._ensure_initialized()

        for i in range(10):
            fingerprint = np.random.randn(20).astype(np.float32)
            storage.store(
                request_id=f"req-{i:03d}",
                step=i,
                fingerprint=fingerprint,
            )

        storage.flush()

        # Count stored
        cursor = storage._conn.execute("SELECT COUNT(*) FROM fingerprints")
        count = cursor.fetchone()[0]
        assert count == 10

    def test_get_fingerprint_count(self, temp_db):
        """Test getting fingerprint count via get_stats()."""
        storage = FingerprintStorage(db_path=temp_db, session_id="test-session")
        storage._ensure_initialized()

        # Store some fingerprints
        for i in range(5):
            storage.store(
                request_id=f"req-{i}",
                step=0,
                fingerprint=np.random.randn(20).astype(np.float32),
            )
        storage.flush()

        stats = storage.get_stats()
        assert stats["session_fingerprints"] == 5

    def test_fingerprint_dimension(self, temp_db):
        """Test fingerprint dimension constant."""
        storage = FingerprintStorage(db_path=temp_db)
        assert storage.FINGERPRINT_DIM == 20

    def test_batch_buffering(self, temp_db):
        """Test batch buffering behavior."""
        storage = FingerprintStorage(db_path=temp_db, session_id="test-session")
        storage._ensure_initialized()
        storage._batch_size = 5

        # Add fewer than batch size
        for i in range(3):
            storage.store(
                request_id=f"req-{i}",
                step=0,
                fingerprint=np.random.randn(20).astype(np.float32),
            )

        # Should be buffered, not yet written
        cursor = storage._conn.execute("SELECT COUNT(*) FROM fingerprints")
        count = cursor.fetchone()[0]
        assert count == 0  # Still buffered

        # Flush to write
        storage.flush()

        cursor = storage._conn.execute("SELECT COUNT(*) FROM fingerprints")
        count = cursor.fetchone()[0]
        assert count == 3

    def test_thread_safety(self, temp_db):
        """Test thread-safe storage."""
        storage = FingerprintStorage(db_path=temp_db, session_id="test-session")
        storage._ensure_initialized()

        errors = []

        def store_thread(thread_id):
            try:
                for i in range(20):
                    storage.store(
                        request_id=f"t{thread_id}-req-{i}",
                        step=i,
                        fingerprint=np.random.randn(20).astype(np.float32),
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_thread, args=(i,)) for i in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        storage.flush()

        assert len(errors) == 0

        cursor = storage._conn.execute("SELECT COUNT(*) FROM fingerprints")
        count = cursor.fetchone()[0]
        assert count == 80  # 4 threads * 20 fingerprints


class TestOptionalDependencies:
    """Tests for optional dependency flags."""

    def test_has_rapids_is_bool(self):
        """Test HAS_RAPIDS is a boolean."""
        assert isinstance(HAS_RAPIDS, bool)

    def test_has_sklearn_is_bool(self):
        """Test HAS_SKLEARN is a boolean."""
        assert isinstance(HAS_SKLEARN, bool)

    def test_has_zmq_is_bool(self):
        """Test HAS_ZMQ is a boolean."""
        assert isinstance(HAS_ZMQ, bool)


class TestFingerprintSerialization:
    """Tests for fingerprint serialization."""

    def test_fingerprint_to_bytes(self):
        """Test converting fingerprint to bytes."""
        fp = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        fp_bytes = fp.tobytes()

        assert isinstance(fp_bytes, bytes)
        assert len(fp_bytes) == 4 * 4  # 4 floats * 4 bytes each

    def test_fingerprint_from_bytes(self):
        """Test converting bytes back to fingerprint."""
        original = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        fp_bytes = original.tobytes()

        restored = np.frombuffer(fp_bytes, dtype=np.float32)

        np.testing.assert_array_almost_equal(original, restored)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
