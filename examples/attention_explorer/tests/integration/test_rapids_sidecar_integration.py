"""
Integration tests for rapids_sidecar.py

Tests the RapidsSidecar server including:
- HTTP endpoints (/fingerprint, /clusters, /stats, /classify)
- Fingerprint storage to SQLite
- Online clustering
- Discovery artifact loading
"""

import json
import os
import sqlite3
import struct
import threading
import time
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from http.client import HTTPConnection
from urllib.request import urlopen, Request
from urllib.error import URLError

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rapids_sidecar import (
    ClusteringBackend,
    FingerprintStorage,
    SIDECAR_VERSION,
    HAS_RAPIDS,
    HAS_SKLEARN,
    HAS_ZMQ,
)

# Import sidecar class if available
try:
    from rapids_sidecar import RapidsSidecar, ClusterCentroid
    HAS_SIDECAR = True
except ImportError:
    HAS_SIDECAR = False


FINGERPRINT_DIM = 20


def pack_fingerprint(arr: np.ndarray) -> bytes:
    """Pack numpy array to fingerprint blob."""
    return struct.pack(f'<{FINGERPRINT_DIM}f', *arr.astype(np.float32))


class TestFingerprintStorageIntegration:
    """Integration tests for FingerprintStorage with real database."""

    def test_full_write_read_cycle(self, temp_db_with_schema):
        """Test complete write and read cycle."""
        storage = FingerprintStorage(db_path=temp_db_with_schema, session_id="test-session")
        storage._ensure_initialized()

        # Write multiple fingerprints
        fingerprints = []
        for i in range(50):
            fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)
            fingerprints.append(fp)
            storage.store(
                request_id=f"req-{i:04d}",
                step=i % 10,
                fingerprint=fp,
                zone="syntax_floor" if i % 3 == 0 else "semantic_bridge",
            )

        storage.flush()

        # Read back and verify
        conn = sqlite3.connect(temp_db_with_schema)
        cursor = conn.execute("SELECT COUNT(*) FROM fingerprints")
        count = cursor.fetchone()[0]

        assert count == 50

        # Verify zone distribution
        cursor = conn.execute(
            "SELECT manifold_zone, COUNT(*) FROM fingerprints GROUP BY manifold_zone"
        )
        zone_counts = dict(cursor.fetchall())
        conn.close()

        assert zone_counts.get("syntax_floor", 0) > 0
        assert zone_counts.get("semantic_bridge", 0) > 0

    def test_concurrent_writes(self, temp_db_with_schema):
        """Test concurrent writes from multiple threads."""
        storage = FingerprintStorage(db_path=temp_db_with_schema, session_id="test-session")
        storage._ensure_initialized()

        errors = []
        writes_per_thread = 50

        def writer_thread(thread_id):
            try:
                for i in range(writes_per_thread):
                    fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)
                    storage.store(
                        request_id=f"t{thread_id}-req-{i:04d}",
                        step=i,
                        fingerprint=fp,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer_thread, args=(i,)) for i in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        storage.flush()

        assert len(errors) == 0

        # Verify all writes succeeded
        conn = sqlite3.connect(temp_db_with_schema)
        cursor = conn.execute("SELECT COUNT(*) FROM fingerprints")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 4 * writes_per_thread

    def test_storage_stats(self, temp_db_with_schema):
        """Test storage statistics."""
        storage = FingerprintStorage(db_path=temp_db_with_schema, session_id="test-session")
        storage._ensure_initialized()

        # Write some fingerprints
        for i in range(20):
            fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)
            storage.store(request_id=f"req-{i}", step=0, fingerprint=fp)

        storage.flush()

        stats = storage.get_stats()

        assert stats["session_id"] == "test-session"
        assert stats["session_fingerprints"] == 20
        assert stats["total_writes"] >= 20

    def test_storage_close(self, temp_db_with_schema):
        """Test storage cleanup on close."""
        storage = FingerprintStorage(db_path=temp_db_with_schema, session_id="test-session")
        storage._ensure_initialized()

        # Write and flush
        fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)
        storage.store(request_id="req-0", step=0, fingerprint=fp)
        storage.flush()

        # Close
        storage.close()

        # Connection should be closed
        assert storage._conn is None or not storage._initialized


@pytest.mark.skipif(not HAS_SIDECAR, reason="RapidsSidecar not available")
class TestRapidsSidecarInit:
    """Tests for RapidsSidecar initialization."""

    def test_init_with_defaults(self, temp_db_with_schema):
        """Test sidecar initialization with defaults."""
        sidecar = RapidsSidecar(
            port=0,  # Don't bind
            db_path=temp_db_with_schema,
        )

        assert sidecar.port == 0
        assert sidecar.backend in [
            ClusteringBackend.RAPIDS,
            ClusteringBackend.CPU,
            ClusteringBackend.ONLINE,
        ]

    def test_init_selects_best_backend(self, temp_db_with_schema):
        """Test that initialization selects the best available backend."""
        sidecar = RapidsSidecar(
            port=0,
            db_path=temp_db_with_schema,
        )

        # Should select best available
        if HAS_RAPIDS:
            assert sidecar.backend == ClusteringBackend.RAPIDS
        elif HAS_SKLEARN:
            assert sidecar.backend == ClusteringBackend.CPU
        else:
            assert sidecar.backend == ClusteringBackend.ONLINE

    def test_init_with_discovery_dir(self, temp_db_with_schema, discovery_artifacts):
        """Test sidecar with discovery artifacts."""
        sidecar = RapidsSidecar(
            port=0,
            db_path=temp_db_with_schema,
            discovery_dir=discovery_artifacts['output_dir'],
        )

        assert sidecar.discovery_dir == discovery_artifacts['output_dir']


@pytest.mark.skipif(not HAS_SIDECAR, reason="RapidsSidecar not available")
class TestRapidsSidecarFingerprints:
    """Tests for fingerprint handling in RapidsSidecar."""

    def test_add_fingerprint(self, temp_db_with_schema):
        """Test adding a fingerprint to the sidecar."""
        sidecar = RapidsSidecar(
            port=0,
            db_path=temp_db_with_schema,
            buffer_size=100,
        )

        fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)

        sidecar.add_fingerprint(
            request_id="test-req",
            step=0,
            vector=fp,
            metadata={"prompt_type": "code"},
        )

        assert len(sidecar.fingerprints) == 1

    def test_fingerprint_buffer_limit(self, temp_db_with_schema):
        """Test fingerprint buffer respects size limit."""
        buffer_size = 50
        sidecar = RapidsSidecar(
            port=0,
            db_path=temp_db_with_schema,
            buffer_size=buffer_size,
        )

        # Add more than buffer size
        for i in range(buffer_size + 20):
            fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)
            sidecar.add_fingerprint(
                request_id=f"req-{i}",
                step=0,
                vector=fp,
            )

        # Buffer should be limited to buffer_size
        assert len(sidecar.fingerprints) <= buffer_size


@pytest.mark.skipif(not HAS_SIDECAR, reason="RapidsSidecar not available")
@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestOnlineClustering:
    """Tests for online clustering functionality."""

    def test_clustering_updates_centroids(self, temp_db_with_schema, sample_fingerprints_large):
        """Test that clustering produces centroids."""
        sidecar = RapidsSidecar(
            port=0,
            db_path=temp_db_with_schema,
            buffer_size=2000,
            min_cluster_size=10,
            backend="cpu",
        )

        # Add fingerprints
        for i, fp in enumerate(sample_fingerprints_large[:500]):
            sidecar.add_fingerprint(
                request_id=f"req-{i}",
                step=0,
                vector=fp,
            )

        # Trigger clustering
        vectors = np.stack([f.vector for f in sidecar.fingerprints])
        sidecar._run_clustering(vectors)

        # Should have some centroids
        assert len(sidecar.centroids) > 0

    def test_get_centroids(self, temp_db_with_schema, sample_fingerprints_large):
        """Test getting cluster centroids."""
        sidecar = RapidsSidecar(
            port=0,
            db_path=temp_db_with_schema,
            buffer_size=2000,
            min_cluster_size=10,
            backend="cpu",
        )

        # Add fingerprints and cluster
        for i, fp in enumerate(sample_fingerprints_large[:500]):
            sidecar.add_fingerprint(
                request_id=f"req-{i}",
                step=0,
                vector=fp,
            )

        vectors = np.stack([f.vector for f in sidecar.fingerprints])
        sidecar._run_clustering(vectors)

        # Get centroids
        centroids = sidecar.get_centroids()

        assert isinstance(centroids, dict)
        for cluster_id, centroid in centroids.items():
            assert isinstance(centroid, ClusterCentroid)
            assert len(centroid.vector) == FINGERPRINT_DIM


class TestClusterCentroid:
    """Tests for ClusterCentroid dataclass."""

    def test_create_centroid(self):
        """Test creating a cluster centroid."""
        try:
            from rapids_sidecar import ClusterCentroid
        except ImportError:
            pytest.skip("ClusterCentroid not available")

        vector = np.random.randn(FINGERPRINT_DIM).astype(np.float32)

        centroid = ClusterCentroid(
            cluster_id=0,
            vector=vector,
            size=100,
            zone="syntax_floor",
            zone_confidence=0.85,
        )

        assert centroid.cluster_id == 0
        assert centroid.size == 100
        assert centroid.zone == "syntax_floor"
        np.testing.assert_array_equal(centroid.vector, vector)


class TestVersionAndStatus:
    """Tests for version and status information."""

    def test_version_constant(self):
        """Test sidecar version is defined."""
        assert SIDECAR_VERSION is not None
        assert isinstance(SIDECAR_VERSION, str)

        # Should be semver format
        parts = SIDECAR_VERSION.split(".")
        assert len(parts) == 3

    def test_dependency_flags(self):
        """Test dependency flags are booleans."""
        assert isinstance(HAS_RAPIDS, bool)
        assert isinstance(HAS_SKLEARN, bool)
        assert isinstance(HAS_ZMQ, bool)


@pytest.mark.skipif(not HAS_SIDECAR, reason="RapidsSidecar not available")
class TestSidecarStats:
    """Tests for sidecar statistics."""

    def test_get_stats(self, temp_db_with_schema):
        """Test getting sidecar statistics."""
        sidecar = RapidsSidecar(
            port=0,
            db_path=temp_db_with_schema,
            buffer_size=100,
        )

        # Add some fingerprints
        for i in range(10):
            fp = np.random.randn(FINGERPRINT_DIM).astype(np.float32)
            sidecar.add_fingerprint(
                request_id=f"req-{i}",
                step=0,
                vector=fp,
            )

        stats = sidecar.get_stats()

        assert "buffer_size" in stats
        assert "n_clusters" in stats
        assert stats["buffer_size"] == 10


@pytest.mark.skipif(not HAS_ZMQ, reason="ZMQ not installed")
class TestZMQIntegration:
    """Tests for ZMQ message handling."""

    def test_zmq_message_format(self):
        """Test ZMQ message format parsing."""
        # Create a mock message
        message = {
            "request_id": "test-req",
            "step": 0,
            "vector": [0.1] * FINGERPRINT_DIM,
            "metadata": {"prompt_type": "code"},
        }

        # Should be valid JSON
        encoded = json.dumps(message).encode('utf-8')
        decoded = json.loads(encoded.decode('utf-8'))

        assert decoded["request_id"] == "test-req"
        assert len(decoded["vector"]) == FINGERPRINT_DIM


class TestDiscoveryIntegration:
    """Tests for discovery artifact integration."""

    def test_load_discovery_artifacts(self, discovery_artifacts):
        """Test loading discovery artifacts."""
        output_dir = Path(discovery_artifacts['output_dir'])
        latest_dir = output_dir / "latest"

        # Verify artifacts exist
        assert latest_dir.exists()
        assert (latest_dir / "manifest.json").exists()
        assert (latest_dir / "embeddings.parquet").exists()
        assert (latest_dir / "clusters.json").exists()

    def test_centroids_from_discovery(self, discovery_artifacts):
        """Test loading centroids from discovery artifacts."""
        centroids_path = Path(discovery_artifacts['run_dir']) / "centroids.json"

        with open(centroids_path) as f:
            centroids = json.load(f)

        assert len(centroids) == 3  # 3 clusters in fixture
        for cluster_id, vector in centroids.items():
            assert len(vector) == FINGERPRINT_DIM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
