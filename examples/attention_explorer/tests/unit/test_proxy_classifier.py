"""
Unit tests for ProxyClassifier

Tests centroid-based classification for online attention routing.
"""

# Add parent to path for imports
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from proxy_classifier import HAS_REQUESTS, HAS_TORCH, CachedCentroid, ProxyClassifier


class TestCachedCentroid:
    """Tests for CachedCentroid dataclass."""

    def test_create_cached_centroid(self):
        """Test creating a cached centroid."""
        centroid = np.array([0.5, 0.3, 0.2, 1.5], dtype=np.float32)

        cached = CachedCentroid(
            cluster_id=3,
            centroid=centroid,
            traits=["local_attention", "structured"],
            sampling_hint={"temperature": 0.3, "top_p": 0.9},
        )

        assert cached.cluster_id == 3
        np.testing.assert_array_equal(cached.centroid, centroid)
        assert "local_attention" in cached.traits
        assert cached.sampling_hint["temperature"] == 0.3

    def test_cached_centroid_empty_traits(self):
        """Test cached centroid with empty traits."""
        cached = CachedCentroid(
            cluster_id=0,
            centroid=np.zeros(4, dtype=np.float32),
            traits=[],
            sampling_hint={},
        )

        assert cached.traits == []
        assert cached.sampling_hint == {}


class TestProxyClassifier:
    """Tests for ProxyClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create a classifier instance."""
        return ProxyClassifier(
            max_distance=2.0,
            sync_interval=30.0,
            feature_dim=20,
        )

    def test_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.max_distance == 2.0
        assert classifier.sync_interval == 30.0
        assert classifier.feature_dim == 20
        assert classifier._classify_count == 0

    def test_default_initialization(self):
        """Test classifier with default values."""
        classifier = ProxyClassifier()

        assert classifier.max_distance == 2.0
        assert classifier.sync_interval == 30.0
        assert classifier.feature_dim == 20

    def test_add_centroid(self, classifier):
        """Test adding a centroid manually."""
        centroid = CachedCentroid(
            cluster_id=1,
            centroid=np.array([0.7, 0.2, 0.1, 1.5] + [0.0] * 16, dtype=np.float32),
            traits=["syntax_floor"],
            sampling_hint={"temperature": 0.2},
        )

        with classifier._centroids_lock:
            classifier._centroids[1] = centroid

        assert 1 in classifier._centroids
        assert classifier._centroids[1].traits == ["syntax_floor"]

    def test_classify_empty_centroids(self, classifier):
        """Test classification with no centroids."""
        fingerprint = np.random.randn(20).astype(np.float32)

        result = classifier.classify(fingerprint)

        # Returns (cluster_id, distance, traits, sampling_hint)
        assert result is not None
        cluster_id, distance, traits, sampling_hint = result
        assert cluster_id is None  # No centroids -> no match
        assert distance == float("inf")
        assert traits == []
        assert sampling_hint == {}

    def test_classify_finds_nearest(self, classifier):
        """Test classification finds nearest centroid."""
        # Add two centroids
        c1 = CachedCentroid(
            cluster_id=0,
            centroid=np.array([1.0] + [0.0] * 19, dtype=np.float32),
            traits=["cluster_a"],
            sampling_hint={"temperature": 0.5},
        )
        c2 = CachedCentroid(
            cluster_id=1,
            centroid=np.array([-1.0] + [0.0] * 19, dtype=np.float32),
            traits=["cluster_b"],
            sampling_hint={"temperature": 0.8},
        )

        with classifier._centroids_lock:
            classifier._centroids[0] = c1
            classifier._centroids[1] = c2

        # Fingerprint close to c1
        fingerprint = np.array([0.9] + [0.0] * 19, dtype=np.float32)
        result = classifier.classify(fingerprint)

        # Returns (cluster_id, distance, traits, sampling_hint)
        assert result is not None
        cluster_id, distance, traits, sampling_hint = result
        # Should be closer to cluster_a (cluster_id=0)
        assert cluster_id == 0
        assert "cluster_a" in traits
        assert sampling_hint.get("temperature") == 0.5

    def test_classify_respects_max_distance(self, classifier):
        """Test that classification respects max_distance threshold."""
        classifier.max_distance = 0.1  # Very small threshold

        c1 = CachedCentroid(
            cluster_id=0,
            centroid=np.array([100.0] + [0.0] * 19, dtype=np.float32),  # Far away
            traits=["far_cluster"],
            sampling_hint={},
        )

        with classifier._centroids_lock:
            classifier._centroids[0] = c1

        # Fingerprint at origin, far from centroid
        fingerprint = np.zeros(20, dtype=np.float32)
        result = classifier.classify(fingerprint)

        # Returns (cluster_id, distance, traits, sampling_hint)
        # Should return None cluster due to distance exceeding max_distance
        cluster_id, distance, traits, sampling_hint = result
        assert cluster_id is None  # Too far away
        assert distance > classifier.max_distance

    def test_thread_safety(self, classifier):
        """Test that classification is thread-safe."""
        # Add a centroid
        c1 = CachedCentroid(
            cluster_id=0,
            centroid=np.zeros(20, dtype=np.float32),
            traits=["test"],
            sampling_hint={},
        )

        with classifier._centroids_lock:
            classifier._centroids[0] = c1

        results = []
        errors = []

        def classify_thread():
            try:
                for _ in range(100):
                    fp = np.random.randn(20).astype(np.float32)
                    result = classifier.classify(fp)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=classify_thread) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 400

    def test_stats_tracking(self, classifier):
        """Test that classification stats are tracked."""
        c1 = CachedCentroid(
            cluster_id=0,
            centroid=np.zeros(20, dtype=np.float32),
            traits=["test"],
            sampling_hint={},
        )

        with classifier._centroids_lock:
            classifier._centroids[0] = c1

        initial_count = classifier._classify_count

        # Classify a few times
        for _ in range(5):
            fp = np.random.randn(20).astype(np.float32)
            classifier.classify(fp)

        assert classifier._classify_count == initial_count + 5


class TestDistanceComputation:
    """Tests for distance computation utilities."""

    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        a = np.array([0, 0, 0], dtype=np.float32)
        b = np.array([3, 4, 0], dtype=np.float32)

        dist = np.linalg.norm(a - b)

        assert dist == pytest.approx(5.0, rel=0.01)

    def test_distance_to_self(self):
        """Test distance to self is zero."""
        a = np.array([1, 2, 3], dtype=np.float32)

        dist = np.linalg.norm(a - a)

        assert dist == 0.0


class TestSyncThread:
    """Tests for centroid sync thread."""

    @pytest.fixture
    def classifier(self):
        return ProxyClassifier(sync_interval=0.1)

    def test_sync_state_initialization(self, classifier):
        """Test sync state is properly initialized."""
        assert classifier._sidecar_url is None
        assert classifier._sync_thread is None
        assert classifier._last_sync == 0

    def test_stop_event_initialized(self, classifier):
        """Test stop event is initialized."""
        assert classifier._stop_event is not None
        assert not classifier._stop_event.is_set()


class TestOptionalDependencies:
    """Tests for optional dependency flags."""

    def test_has_requests_is_bool(self):
        """Test HAS_REQUESTS is a boolean."""
        assert isinstance(HAS_REQUESTS, bool)

    def test_has_torch_is_bool(self):
        """Test HAS_TORCH is a boolean."""
        assert isinstance(HAS_TORCH, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
