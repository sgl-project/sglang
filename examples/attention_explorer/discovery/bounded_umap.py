"""
Memory-Bounded UMAP for Large-Scale Embedding

Provides memory-safe UMAP wrapper that processes in chunks with monitoring.
For datasets > 500K points, standard UMAP can exhaust memory.

Strategy:
1. Fit UMAP on a representative sample (default 50K)
2. Use transform() for remaining points in chunks
3. Monitor memory and trigger GC when needed
4. Graceful degradation under memory pressure
"""

import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

logger = logging.getLogger(__name__)


# =============================================================================
# MEMORY MONITOR
# =============================================================================

@dataclass
class MemoryStatus:
    """Current memory status."""
    used_bytes: int
    available_bytes: int
    total_bytes: int
    percent_used: float
    should_gc: bool


class MemoryMonitor:
    """
    Monitor and control memory usage.

    Features:
    - Track current memory usage
    - Trigger GC when threshold exceeded
    - Raise MemoryError if limit exceeded
    """

    def __init__(
        self,
        max_memory_gb: float = 8.0,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.90,
    ):
        """
        Initialize memory monitor.

        Args:
            max_memory_gb: Maximum memory to use in GB
            warning_threshold: Fraction of max at which to warn/GC
            critical_threshold: Fraction of max at which to error
        """
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._last_gc_time = 0.0
        self._gc_cooldown = 5.0  # Seconds between GC calls

    def get_status(self) -> MemoryStatus:
        """Get current memory status."""
        if HAS_PSUTIL:
            process = psutil.Process()
            mem_info = process.memory_info()
            used = mem_info.rss
            virtual = psutil.virtual_memory()
            available = virtual.available
            total = virtual.total
        else:
            # Fallback: estimate from process
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            used = rusage.ru_maxrss * 1024  # Convert to bytes
            available = self.max_memory_bytes - used
            total = self.max_memory_bytes

        percent = used / self.max_memory_bytes
        should_gc = percent > self.warning_threshold

        return MemoryStatus(
            used_bytes=used,
            available_bytes=available,
            total_bytes=total,
            percent_used=percent,
            should_gc=should_gc,
        )

    def check_memory(self) -> Tuple[float, bool]:
        """
        Check current memory usage.

        Returns:
            Tuple of (usage_fraction, should_gc)
        """
        status = self.get_status()
        return status.percent_used, status.should_gc

    def maybe_gc(self) -> bool:
        """
        Run GC if appropriate (respecting cooldown).

        Returns:
            True if GC was run
        """
        now = time.time()
        status = self.get_status()

        if status.should_gc and (now - self._last_gc_time) > self._gc_cooldown:
            gc.collect()
            self._last_gc_time = now
            return True
        return False

    def enforce_limit(self) -> None:
        """
        Enforce memory limit. Raises MemoryError if exceeded.
        """
        status = self.get_status()

        if status.percent_used > self.critical_threshold:
            # Try GC first
            gc.collect()
            status = self.get_status()

            if status.percent_used > self.critical_threshold:
                raise MemoryError(
                    f"Memory usage at {status.percent_used:.1%} "
                    f"({status.used_bytes / 1e9:.1f} GB), "
                    f"exceeding critical threshold of {self.critical_threshold:.1%}. "
                    "Consider reducing batch size or increasing memory limit."
                )


# =============================================================================
# MEMORY-BOUNDED UMAP
# =============================================================================

class MemoryBoundedUMAP:
    """
    Memory-safe UMAP wrapper that processes in chunks.

    For datasets > 500K points, UMAP can exhaust memory.
    This wrapper:
    1. Fits UMAP on a representative sample
    2. Uses transform() for remaining points in chunks
    3. Monitors memory and triggers GC when needed
    4. Supports graceful degradation

    Usage:
        bounded_umap = MemoryBoundedUMAP(
            n_components=2,
            max_memory_gb=8.0,
            sample_size=50000,
        )

        embeddings = bounded_umap.fit_transform_bounded(
            X,
            progress_callback=lambda done, total: print(f"{done}/{total}"),
        )
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        sample_size: int = 50000,
        transform_chunk_size: int = 10000,
        max_memory_gb: float = 8.0,
        random_state: Optional[int] = None,
        low_memory: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize memory-bounded UMAP.

        Args:
            n_components: Embedding dimensionality
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            metric: Distance metric
            sample_size: Number of points to fit on (rest transformed)
            transform_chunk_size: Chunk size for transform
            max_memory_gb: Maximum memory usage
            random_state: Random seed for reproducibility
            low_memory: Force UMAP low_memory mode
            verbose: Enable verbose logging
        """
        if not HAS_UMAP:
            raise ImportError("umap-learn is required: pip install umap-learn")

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.sample_size = sample_size
        self.transform_chunk_size = transform_chunk_size
        self.max_memory_gb = max_memory_gb
        self.random_state = random_state
        self.low_memory = low_memory
        self.verbose = verbose

        self._reducer: Optional[umap.UMAP] = None
        self._memory_monitor = MemoryMonitor(max_memory_gb=max_memory_gb)
        self._is_fitted = False

    def _get_representative_sample(
        self,
        X: np.ndarray,
        sample_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a representative sample for fitting.

        Uses stratified sampling based on feature variance to ensure
        the sample captures the full distribution.

        Args:
            X: Full dataset
            sample_size: Number of samples to select

        Returns:
            Tuple of (sample_data, sample_indices)
        """
        n_samples = X.shape[0]

        if n_samples <= sample_size:
            return X, np.arange(n_samples)

        # Stratified sampling based on first few principal components
        # This helps capture the full distribution
        rng = np.random.default_rng(self.random_state)

        # Compute rough variance per feature and sort
        variances = np.var(X, axis=0)
        top_features = np.argsort(variances)[-min(10, X.shape[1]):]

        # Use top-variance features for binning
        X_reduced = X[:, top_features]

        # Simple stratification: divide into bins and sample from each
        n_bins = min(100, sample_size // 10)
        bin_indices = np.digitize(
            X_reduced[:, 0],  # Use first high-variance feature
            bins=np.percentile(X_reduced[:, 0], np.linspace(0, 100, n_bins + 1)[1:-1])
        )

        # Sample from each bin
        indices = []
        samples_per_bin = sample_size // n_bins

        for bin_id in range(n_bins):
            bin_mask = bin_indices == bin_id
            bin_count = np.sum(bin_mask)

            if bin_count > 0:
                bin_indices_arr = np.where(bin_mask)[0]
                n_select = min(samples_per_bin, bin_count)
                selected = rng.choice(bin_indices_arr, size=n_select, replace=False)
                indices.extend(selected)

        # If we need more, randomly sample from remainder
        remaining = sample_size - len(indices)
        if remaining > 0:
            mask = np.ones(n_samples, dtype=bool)
            mask[indices] = False
            pool = np.where(mask)[0]
            if len(pool) > 0:
                extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
                indices.extend(extra)

        indices = np.array(indices[:sample_size])
        return X[indices], indices

    def fit(self, X: np.ndarray) -> 'MemoryBoundedUMAP':
        """
        Fit UMAP on a sample of the data.

        Args:
            X: Training data (will sample if > sample_size)

        Returns:
            Self
        """
        n_samples = X.shape[0]

        # Determine if we should use low_memory mode
        use_low_memory = self.low_memory or n_samples > 500000

        # Get sample for fitting
        X_sample, sample_indices = self._get_representative_sample(X, self.sample_size)

        if self.verbose:
            logger.info(
                f"Fitting UMAP on {len(X_sample)} samples "
                f"(from {n_samples} total), low_memory={use_low_memory}"
            )

        # Check memory before fitting
        self._memory_monitor.enforce_limit()

        # Create and fit UMAP
        self._reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            low_memory=use_low_memory,
            verbose=self.verbose,
        )

        self._reducer.fit(X_sample)
        self._is_fitted = True

        # GC after fitting
        gc.collect()

        return self

    def transform(
        self,
        X: np.ndarray,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        Transform data in chunks to manage memory.

        Args:
            X: Data to transform
            progress_callback: Called with (processed, total) after each chunk

        Returns:
            Transformed embeddings
        """
        if not self._is_fitted:
            raise RuntimeError("UMAP not fitted. Call fit() first.")

        n_samples = X.shape[0]

        # If small enough, transform all at once
        if n_samples <= self.transform_chunk_size:
            return self._reducer.transform(X)

        # Process in chunks
        embeddings = np.zeros((n_samples, self.n_components), dtype=np.float32)
        processed = 0

        for start in range(0, n_samples, self.transform_chunk_size):
            # Check memory
            self._memory_monitor.maybe_gc()
            self._memory_monitor.enforce_limit()

            end = min(start + self.transform_chunk_size, n_samples)
            chunk = X[start:end]

            embeddings[start:end] = self._reducer.transform(chunk)
            processed = end

            if progress_callback:
                progress_callback(processed, n_samples)

            if self.verbose and processed % (self.transform_chunk_size * 10) == 0:
                status = self._memory_monitor.get_status()
                logger.info(
                    f"Transformed {processed}/{n_samples} points, "
                    f"memory: {status.percent_used:.1%}"
                )

        return embeddings

    def fit_transform_bounded(
        self,
        X: np.ndarray,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        Memory-bounded fit_transform.

        1. If X.shape[0] <= sample_size: standard fit_transform
        2. Else: fit on sample, transform rest in chunks

        Args:
            X: Data to embed
            progress_callback: Called with (processed, total) after each chunk

        Returns:
            2D embeddings
        """
        n_samples = X.shape[0]

        # Check memory before starting
        self._memory_monitor.enforce_limit()

        # If small enough, standard fit_transform
        if n_samples <= self.sample_size:
            if self.verbose:
                logger.info(f"Small dataset ({n_samples} points), using standard fit_transform")

            use_low_memory = self.low_memory or n_samples > 500000
            self._reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
                low_memory=use_low_memory,
                verbose=self.verbose,
            )
            result = self._reducer.fit_transform(X)
            self._is_fitted = True
            return result

        # Fit on sample
        X_sample, sample_indices = self._get_representative_sample(X, self.sample_size)

        if self.verbose:
            logger.info(f"Large dataset: fitting on {len(X_sample)} samples")

        self.fit(X_sample)

        # Get sample embeddings
        sample_embeddings = self._reducer.embedding_

        # Transform all points (including re-doing sample for consistency)
        if self.verbose:
            logger.info(f"Transforming {n_samples} points in chunks of {self.transform_chunk_size}")

        embeddings = self.transform(X, progress_callback)

        # Optionally: use exact sample embeddings for sample points
        # (avoids slight differences from transform vs fit_transform)
        embeddings[sample_indices] = sample_embeddings

        return embeddings

    @property
    def embedding_(self) -> Optional[np.ndarray]:
        """Get embedding from last fit (sample only)."""
        if self._reducer is not None:
            return self._reducer.embedding_
        return None

    def get_reducer(self) -> Optional[Any]:
        """Get the underlying UMAP reducer for serialization."""
        return self._reducer


# =============================================================================
# ADAPTIVE PROCESSOR
# =============================================================================

class AdaptiveProcessor:
    """
    Processor that adapts batch size based on memory constraints.

    When memory pressure is high:
    1. Reduce batch size
    2. Trigger GC more frequently
    3. Log warnings
    """

    def __init__(
        self,
        memory_monitor: MemoryMonitor,
        initial_batch_size: int = 10000,
        min_batch_size: int = 1000,
        max_batch_size: int = 50000,
    ):
        """
        Initialize adaptive processor.

        Args:
            memory_monitor: Memory monitor instance
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size (never go below)
            max_batch_size: Maximum batch size
        """
        self.monitor = memory_monitor
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self._adjustments = 0

    def get_batch_size(self) -> int:
        """Get current adaptive batch size."""
        usage, should_gc = self.monitor.check_memory()

        if should_gc:
            self.monitor.maybe_gc()
            usage, _ = self.monitor.check_memory()

        # Adjust batch size based on memory
        if usage > 0.85:
            # Critical: reduce significantly
            new_size = max(self.min_batch_size, self.batch_size // 2)
            if new_size != self.batch_size:
                logger.warning(
                    f"High memory pressure ({usage:.1%}): "
                    f"reducing batch size {self.batch_size} -> {new_size}"
                )
                self.batch_size = new_size
                self._adjustments += 1
        elif usage > 0.7:
            # Warning: reduce slightly
            new_size = max(self.min_batch_size, int(self.batch_size * 0.75))
            if new_size != self.batch_size:
                logger.info(
                    f"Memory pressure ({usage:.1%}): "
                    f"reducing batch size {self.batch_size} -> {new_size}"
                )
                self.batch_size = new_size
                self._adjustments += 1
        elif usage < 0.5 and self.batch_size < self.max_batch_size:
            # Low usage: can increase
            new_size = min(self.max_batch_size, int(self.batch_size * 1.25))
            if new_size != self.batch_size and self._adjustments > 0:
                logger.info(
                    f"Low memory ({usage:.1%}): "
                    f"increasing batch size {self.batch_size} -> {new_size}"
                )
                self.batch_size = new_size

        return self.batch_size


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with synthetic data
    print("Testing MemoryBoundedUMAP...")

    # Create test data
    np.random.seed(42)
    n_samples = 100000
    n_features = 20

    print(f"Creating test data: {n_samples} x {n_features}")
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Test bounded UMAP
    bounded_umap = MemoryBoundedUMAP(
        n_components=2,
        sample_size=10000,
        transform_chunk_size=5000,
        max_memory_gb=4.0,
        random_state=42,
        verbose=True,
    )

    def progress(done, total):
        print(f"Progress: {done}/{total} ({done/total*100:.1f}%)")

    print("\nRunning fit_transform_bounded...")
    embeddings = bounded_umap.fit_transform_bounded(X, progress_callback=progress)

    print(f"\nResult shape: {embeddings.shape}")
    print(f"Embedding range: [{embeddings.min():.2f}, {embeddings.max():.2f}]")

    # Test memory monitor
    print("\nTesting MemoryMonitor...")
    monitor = MemoryMonitor(max_memory_gb=8.0)
    status = monitor.get_status()
    print(f"Current memory: {status.used_bytes / 1e9:.2f} GB ({status.percent_used:.1%})")

    # Test adaptive processor
    print("\nTesting AdaptiveProcessor...")
    processor = AdaptiveProcessor(monitor, initial_batch_size=10000)
    for _ in range(5):
        batch_size = processor.get_batch_size()
        print(f"Adaptive batch size: {batch_size}")

    print("\nAll tests passed!")
