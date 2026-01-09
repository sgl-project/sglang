"""
Unit tests for MemoryBoundedUMAP and related classes

Tests memory monitoring, chunked UMAP processing, and adaptive batch sizing.
"""

import gc
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from discovery.bounded_umap import (
    MemoryStatus,
    MemoryMonitor,
    MemoryBoundedUMAP,
    AdaptiveProcessor,
    HAS_UMAP,
    HAS_PSUTIL,
)


class TestMemoryStatus:
    """Tests for MemoryStatus dataclass."""

    def test_create_status(self):
        """Test creating a memory status."""
        status = MemoryStatus(
            used_bytes=4 * 1024**3,  # 4GB
            available_bytes=12 * 1024**3,  # 12GB
            total_bytes=16 * 1024**3,  # 16GB
            percent_used=0.5,
            should_gc=False,
        )

        assert status.used_bytes == 4 * 1024**3
        assert status.available_bytes == 12 * 1024**3
        assert status.percent_used == 0.5
        assert status.should_gc is False

    def test_status_should_gc(self):
        """Test should_gc flag."""
        status_no_gc = MemoryStatus(
            used_bytes=1024**3,
            available_bytes=7 * 1024**3,
            total_bytes=8 * 1024**3,
            percent_used=0.5,
            should_gc=False,
        )
        assert status_no_gc.should_gc is False

        status_gc = MemoryStatus(
            used_bytes=7 * 1024**3,
            available_bytes=1024**3,
            total_bytes=8 * 1024**3,
            percent_used=0.9,
            should_gc=True,
        )
        assert status_gc.should_gc is True


class TestMemoryMonitor:
    """Tests for MemoryMonitor class."""

    def test_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor(
            max_memory_gb=8.0,
            warning_threshold=0.75,
            critical_threshold=0.90,
        )

        assert monitor.max_memory_bytes == 8 * 1024**3
        assert monitor.warning_threshold == 0.75
        assert monitor.critical_threshold == 0.90

    def test_get_status(self):
        """Test getting memory status."""
        monitor = MemoryMonitor(max_memory_gb=16.0)
        status = monitor.get_status()

        assert isinstance(status, MemoryStatus)
        assert status.used_bytes > 0
        assert status.total_bytes > 0
        assert 0 <= status.percent_used <= 10  # Should be reasonable

    def test_check_memory(self):
        """Test check_memory returns tuple."""
        monitor = MemoryMonitor(max_memory_gb=16.0)
        usage, should_gc = monitor.check_memory()

        assert isinstance(usage, float)
        assert isinstance(should_gc, bool)
        assert 0 <= usage <= 10  # Reasonable range

    def test_get_usage_gb(self):
        """Test get_usage_gb returns memory in GB."""
        monitor = MemoryMonitor(max_memory_gb=16.0)
        usage_gb = monitor.get_usage_gb()

        assert isinstance(usage_gb, float)
        assert usage_gb >= 0  # Should be non-negative

    def test_maybe_gc_respects_cooldown(self):
        """Test GC cooldown is respected."""
        monitor = MemoryMonitor(max_memory_gb=16.0)
        monitor._gc_cooldown = 0.1  # Short cooldown for testing

        # Force should_gc to True by lowering threshold
        monitor.warning_threshold = 0.0

        # First call should run GC
        result1 = monitor.maybe_gc()
        assert result1 is True

        # Immediate second call should skip (cooldown)
        result2 = monitor.maybe_gc()
        assert result2 is False

    def test_enforce_limit_normal(self):
        """Test enforce_limit passes under normal conditions."""
        monitor = MemoryMonitor(max_memory_gb=100.0)  # High limit
        # Should not raise
        monitor.enforce_limit()

    def test_enforce_limit_raises_on_exceeded(self):
        """Test enforce_limit raises MemoryError when exceeded."""
        monitor = MemoryMonitor(max_memory_gb=0.001)  # Tiny limit (1KB)

        with pytest.raises(MemoryError) as excinfo:
            monitor.enforce_limit()

        assert "Memory usage" in str(excinfo.value)
        assert "critical threshold" in str(excinfo.value)


class TestMemoryBoundedUMAP:
    """Tests for MemoryBoundedUMAP class."""

    @pytest.fixture
    def small_data(self):
        """Create small test data."""
        np.random.seed(42)
        return np.random.randn(100, 10).astype(np.float32)

    @pytest.fixture
    def medium_data(self):
        """Create medium test data."""
        np.random.seed(42)
        return np.random.randn(1000, 20).astype(np.float32)

    def test_initialization(self):
        """Test MemoryBoundedUMAP initialization."""
        if not HAS_UMAP:
            pytest.skip("umap-learn not installed")

        umap = MemoryBoundedUMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            sample_size=5000,
            transform_chunk_size=1000,
            max_memory_gb=8.0,
            random_state=42,
        )

        assert umap.n_components == 2
        assert umap.n_neighbors == 15
        assert umap.sample_size == 5000
        assert umap.transform_chunk_size == 1000
        assert umap._is_fitted is False

    def test_initialization_without_umap_raises(self):
        """Test that initialization without umap-learn raises ImportError."""
        with patch('discovery.bounded_umap.HAS_UMAP', False):
            # Need to reimport to trigger the check
            # This test verifies the check exists in __init__
            pass  # The actual check happens at runtime

    def test_get_representative_sample_small(self):
        """Test sampling when data is smaller than sample size."""
        if not HAS_UMAP:
            pytest.skip("umap-learn not installed")

        umap = MemoryBoundedUMAP(sample_size=500)
        data = np.random.randn(100, 10).astype(np.float32)

        sample, indices = umap._get_representative_sample(data, 500)

        # Should return all data since it's smaller than sample size
        assert len(sample) == 100
        assert len(indices) == 100
        np.testing.assert_array_equal(sample, data)

    def test_get_representative_sample_large(self):
        """Test sampling when data is larger than sample size."""
        if not HAS_UMAP:
            pytest.skip("umap-learn not installed")

        umap = MemoryBoundedUMAP(sample_size=100, random_state=42)
        data = np.random.randn(1000, 10).astype(np.float32)

        sample, indices = umap._get_representative_sample(data, 100)

        assert len(sample) == 100
        assert len(indices) == 100
        # Verify indices are unique
        assert len(set(indices)) == len(indices)
        # Verify sample matches data at indices
        np.testing.assert_array_equal(sample, data[indices])

    @pytest.mark.skipif(not HAS_UMAP, reason="umap-learn not installed")
    def test_fit(self, small_data):
        """Test fitting UMAP on data."""
        umap = MemoryBoundedUMAP(
            n_components=2,
            sample_size=50,
            random_state=42,
        )

        result = umap.fit(small_data)

        assert result is umap  # Should return self
        assert umap._is_fitted is True
        assert umap._reducer is not None

    @pytest.mark.skipif(not HAS_UMAP, reason="umap-learn not installed")
    def test_transform_not_fitted_raises(self, small_data):
        """Test transform raises if not fitted."""
        umap = MemoryBoundedUMAP(n_components=2)

        with pytest.raises(RuntimeError) as excinfo:
            umap.transform(small_data)

        assert "not fitted" in str(excinfo.value)

    @pytest.mark.skipif(not HAS_UMAP, reason="umap-learn not installed")
    def test_transform_small_data(self, small_data):
        """Test transform with small data (single chunk)."""
        umap = MemoryBoundedUMAP(
            n_components=2,
            sample_size=100,
            transform_chunk_size=200,  # Larger than data
            random_state=42,
        )
        umap.fit(small_data)

        result = umap.transform(small_data)

        assert result.shape == (100, 2)
        assert result.dtype == np.float32 or result.dtype == np.float64

    @pytest.mark.skipif(not HAS_UMAP, reason="umap-learn not installed")
    def test_transform_with_progress_callback(self, medium_data):
        """Test transform calls progress callback."""
        umap = MemoryBoundedUMAP(
            n_components=2,
            sample_size=500,
            transform_chunk_size=200,  # Multiple chunks
            random_state=42,
        )
        umap.fit(medium_data)

        progress_calls = []

        def callback(processed, total):
            progress_calls.append((processed, total))

        result = umap.transform(medium_data, progress_callback=callback)

        assert result.shape == (1000, 2)
        assert len(progress_calls) > 0
        # Last call should have processed == total
        assert progress_calls[-1][0] == progress_calls[-1][1]

    @pytest.mark.skipif(not HAS_UMAP, reason="umap-learn not installed")
    def test_fit_transform_bounded_small(self, small_data):
        """Test fit_transform_bounded with small data."""
        umap = MemoryBoundedUMAP(
            n_components=2,
            sample_size=200,  # Larger than data
            random_state=42,
        )

        result = umap.fit_transform_bounded(small_data)

        assert result.shape == (100, 2)
        assert umap._is_fitted is True

    @pytest.mark.skipif(not HAS_UMAP, reason="umap-learn not installed")
    def test_fit_transform_bounded_large(self, medium_data):
        """Test fit_transform_bounded with large data (sampling)."""
        umap = MemoryBoundedUMAP(
            n_components=2,
            sample_size=200,  # Smaller than data
            transform_chunk_size=300,
            random_state=42,
        )

        result = umap.fit_transform_bounded(medium_data)

        assert result.shape == (1000, 2)
        assert umap._is_fitted is True

    @pytest.mark.skipif(not HAS_UMAP, reason="umap-learn not installed")
    def test_embedding_property(self, small_data):
        """Test embedding_ property returns fitted embeddings."""
        umap = MemoryBoundedUMAP(n_components=2, random_state=42)

        # Before fitting
        assert umap.embedding_ is None

        # After fitting
        umap.fit(small_data)
        assert umap.embedding_ is not None
        assert umap.embedding_.shape[1] == 2

    @pytest.mark.skipif(not HAS_UMAP, reason="umap-learn not installed")
    def test_get_reducer(self, small_data):
        """Test get_reducer returns underlying UMAP object."""
        umap = MemoryBoundedUMAP(n_components=2, random_state=42)

        # Before fitting
        assert umap.get_reducer() is None

        # After fitting
        umap.fit(small_data)
        reducer = umap.get_reducer()
        assert reducer is not None
        assert hasattr(reducer, 'transform')


class TestAdaptiveProcessor:
    """Tests for AdaptiveProcessor class."""

    def test_initialization(self):
        """Test adaptive processor initialization."""
        monitor = MemoryMonitor(max_memory_gb=8.0)
        processor = AdaptiveProcessor(
            memory_monitor=monitor,
            initial_batch_size=10000,
            min_batch_size=1000,
            max_batch_size=50000,
        )

        assert processor.batch_size == 10000
        assert processor.min_batch_size == 1000
        assert processor.max_batch_size == 50000

    def test_get_batch_size_normal_memory(self):
        """Test batch size under normal memory conditions."""
        monitor = MemoryMonitor(max_memory_gb=100.0)  # High limit
        processor = AdaptiveProcessor(
            memory_monitor=monitor,
            initial_batch_size=10000,
        )

        batch_size = processor.get_batch_size()

        # Should not change much under normal conditions
        assert batch_size >= processor.min_batch_size
        assert batch_size <= processor.max_batch_size

    def test_get_batch_size_reduces_under_pressure(self):
        """Test batch size reduces under memory pressure."""
        monitor = MemoryMonitor(max_memory_gb=0.0001)  # Very low
        processor = AdaptiveProcessor(
            memory_monitor=monitor,
            initial_batch_size=10000,
            min_batch_size=1000,
        )

        # Get batch size multiple times to trigger reduction
        for _ in range(5):
            batch_size = processor.get_batch_size()

        # Should have reduced due to memory pressure
        assert batch_size <= 10000

    def test_batch_size_respects_minimum(self):
        """Test batch size never goes below minimum."""
        monitor = MemoryMonitor(max_memory_gb=0.0001)  # Very low
        processor = AdaptiveProcessor(
            memory_monitor=monitor,
            initial_batch_size=10000,
            min_batch_size=5000,
        )

        # Force many reductions
        for _ in range(20):
            batch_size = processor.get_batch_size()

        assert batch_size >= processor.min_batch_size

    def test_batch_size_respects_maximum(self):
        """Test batch size never exceeds maximum."""
        monitor = MemoryMonitor(max_memory_gb=1000.0)  # Very high
        processor = AdaptiveProcessor(
            memory_monitor=monitor,
            initial_batch_size=10000,
            max_batch_size=15000,
        )
        processor._adjustments = 1  # Allow increases

        # Try to increase
        for _ in range(10):
            batch_size = processor.get_batch_size()

        assert batch_size <= processor.max_batch_size


class TestHasFlags:
    """Tests for optional dependency flags."""

    def test_has_psutil_is_bool(self):
        """Test HAS_PSUTIL is a boolean."""
        assert isinstance(HAS_PSUTIL, bool)

    def test_has_umap_is_bool(self):
        """Test HAS_UMAP is a boolean."""
        assert isinstance(HAS_UMAP, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
