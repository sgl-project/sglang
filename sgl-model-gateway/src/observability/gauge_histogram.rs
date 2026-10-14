//! Non-cumulative gauge histogram for Grafana heatmap visualization.
//!
//! Unlike Prometheus Histogram which uses cumulative `le` buckets, this emits
//! non-cumulative bucket counts with `(gt, le]` ranges suitable for heatmaps.
//!
//! # Design: True Zero-Allocation Hot Path
//!
//! The key insight is that `gauge!` returns a `Gauge` handle that can be stored.
//! By pre-registering all gauge handles at startup, the hot path becomes just
//! N+1 atomic `gauge.set()` calls with zero allocations.
//!
//! # Performance Characteristics
//!
//! Setup (once per label combination):
//! - `register()`: N+1 gauge registrations, N+1 String allocations for gt/le
//!
//! Hot path (`set_counts()`):
//! - **Zero heap allocations**
//! - **Zero key lookups** (handles are pre-registered)
//! - N+1 atomic `gauge.set()` calls
//!
//! # Example
//!
//! ```ignore
//! use crate::observability::gauge_histogram::{BucketBounds, GaugeHistogramVec};
//!
//! // Define at module level
//! static BOUNDS: BucketBounds<10> = BucketBounds::new([1, 2, 3, 5, 7, 10, 20, 50, 100, 200]);
//! static HISTOGRAM: GaugeHistogramVec<10> = GaugeHistogramVec::new("smg_request_dist", &BOUNDS);
//!
//! // At startup: register for each label combination
//! let handle = HISTOGRAM.register(&[("router", "round_robin"), ("model", "llama")]);
//!
//! // Pre-allocate counts buffer
//! let mut counts = vec![0usize; BOUNDS.bucket_count()];
//!
//! // Hot path: TRUE zero allocation
//! fn update(handle: &GaugeHistogramHandle, counts: &mut [usize], observations: &[u64]) {
//!     BOUNDS.compute_counts_into(counts, observations);
//!     handle.set_counts(counts);  // Just N+1 atomic gauge.set() calls!
//! }
//! ```

use std::sync::Arc;

use dashmap::DashMap;
use metrics::{gauge, Label};

// =============================================================================
// BUCKET BOUNDS
// =============================================================================

/// Static bucket boundary configuration.
///
/// Uses const generics to define bucket bounds at compile time with validation.
/// The bounds define `N` upper limits, creating `N + 1` buckets:
/// `(0, b[0]], (b[0], b[1]], ..., (b[N-1], +Inf]`.
#[derive(Debug)]
pub struct BucketBounds<const N: usize> {
    bounds: [u64; N],
}

impl<const N: usize> BucketBounds<N> {
    /// Create new bucket bounds from a sorted array of upper limits.
    ///
    /// # Panics
    ///
    /// Panics at compile time (in const context) or runtime if bounds are not
    /// strictly ascending.
    #[must_use]
    pub const fn new(bounds: [u64; N]) -> Self {
        let mut i = 1;
        while i < N {
            assert!(
                bounds[i] > bounds[i - 1],
                "bucket bounds must be strictly ascending"
            );
            i += 1;
        }
        Self { bounds }
    }

    /// Returns the number of buckets (one more than the number of bounds).
    #[inline]
    #[must_use]
    pub const fn bucket_count(&self) -> usize {
        N + 1
    }

    /// Returns the number of bounds.
    #[inline]
    #[must_use]
    pub const fn bound_count(&self) -> usize {
        N
    }

    /// Get the bounds array.
    #[inline]
    #[must_use]
    pub const fn bounds(&self) -> &[u64; N] {
        &self.bounds
    }

    /// Find the bucket index for a value. O(log N).
    #[inline]
    #[must_use]
    pub fn bucket_index(&self, value: u64) -> usize {
        self.bounds.partition_point(|&bound| bound < value)
    }

    /// Get the upper bound for a bucket index, or None for the +Inf bucket.
    #[inline]
    #[must_use]
    pub const fn upper_bound(&self, idx: usize) -> Option<u64> {
        if idx < N {
            Some(self.bounds[idx])
        } else {
            None
        }
    }

    /// Get the lower bound for a bucket index (0 for the first bucket).
    #[inline]
    #[must_use]
    pub const fn lower_bound(&self, idx: usize) -> u64 {
        if idx == 0 {
            0
        } else {
            self.bounds[idx - 1]
        }
    }

    /// Compute bucket counts into a pre-allocated buffer. **Zero allocation.**
    ///
    /// # Panics
    ///
    /// Panics if `counts.len() < bucket_count()`.
    #[inline]
    pub fn compute_counts_into(&self, counts: &mut [usize], observations: &[u64]) {
        debug_assert!(
            counts.len() >= self.bucket_count(),
            "counts buffer too small"
        );
        counts[..self.bucket_count()].fill(0);
        for &value in observations {
            let idx = self.bucket_index(value);
            counts[idx] += 1;
        }
    }

    /// Compute bucket counts, allocating a new Vec.
    ///
    /// Prefer `compute_counts_into` in hot paths to avoid allocation.
    #[must_use]
    pub fn compute_counts(&self, observations: &[u64]) -> Vec<usize> {
        let mut counts = vec![0usize; self.bucket_count()];
        self.compute_counts_into(&mut counts, observations);
        counts
    }
}

// =============================================================================
// GAUGE HISTOGRAM HANDLE (pre-registered, zero-alloc hot path)
// =============================================================================

/// Pre-registered gauge handles for a histogram with specific labels.
///
/// This is what you use in the hot path. Calling `set_counts()` does only
/// N+1 atomic `gauge.set()` operations - zero allocations, zero lookups.
#[derive(Clone)]
pub struct GaugeHistogramHandle {
    gauges: Vec<metrics::Gauge>,
}

impl GaugeHistogramHandle {
    /// Set bucket counts. **TRUE zero allocation.**
    ///
    /// Just N+1 atomic `gauge.set()` calls - no key lookup, no allocation.
    #[inline]
    pub fn set_counts(&self, counts: &[usize]) {
        debug_assert_eq!(
            counts.len(),
            self.gauges.len(),
            "counts length must match bucket count"
        );
        for (gauge, &count) in self.gauges.iter().zip(counts.iter()) {
            gauge.set(count as f64);
        }
    }

    /// Number of buckets.
    #[inline]
    pub fn bucket_count(&self) -> usize {
        self.gauges.len()
    }

    /// Zero out all gauges. **Zero allocation.**
    #[inline]
    pub fn zero_counts(&self) {
        for gauge in &self.gauges {
            gauge.set(0.0);
        }
    }
}

// =============================================================================
// GAUGE HISTOGRAM VEC (factory for registering handles)
// =============================================================================

/// Factory for creating pre-registered histogram handles.
///
/// Define as a static constant, then call `register()` for each label combination
/// you need. The returned `GaugeHistogramHandle` provides zero-allocation updates.
#[derive(Debug)]
pub struct GaugeHistogramVec<const N: usize> {
    name: &'static str,
    bounds: &'static BucketBounds<N>,
}

impl<const N: usize> GaugeHistogramVec<N> {
    /// Create a new gauge histogram factory.
    ///
    /// This just stores the name and bounds - no allocation or registration yet.
    #[must_use]
    pub const fn new(name: &'static str, bounds: &'static BucketBounds<N>) -> Self {
        Self { name, bounds }
    }

    #[inline]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    #[inline]
    pub const fn bounds(&self) -> &BucketBounds<N> {
        self.bounds
    }

    /// Register gauges for a specific label combination.
    ///
    /// Call this once per unique label combination (at startup or when first seen).
    /// The returned handle can be cloned cheaply (just Arc clones internally).
    ///
    /// # Arguments
    ///
    /// - `labels`: Static key-value label pairs for this histogram instance
    ///
    /// # Example
    ///
    /// ```ignore
    /// let handle = HISTOGRAM.register(&[("router", "round_robin"), ("model", "llama")]);
    /// ```
    pub fn register(&self, labels: &[(&'static str, &str)]) -> GaugeHistogramHandle {
        let bucket_count = self.bounds.bucket_count();
        let mut gauges = Vec::with_capacity(bucket_count);

        for i in 0..bucket_count {
            // Build gt/le labels for this bucket
            let gt_str = if i == 0 {
                "0".to_string()
            } else {
                self.bounds.bounds[i - 1].to_string()
            };

            let le_str = if i < N {
                self.bounds.bounds[i].to_string()
            } else {
                "+Inf".to_string()
            };

            // Build complete label set
            let mut all_labels: Vec<Label> = Vec::with_capacity(labels.len() + 2);
            for &(k, v) in labels {
                all_labels.push(Label::new(k, v.to_string()));
            }
            all_labels.push(Label::new("gt", gt_str));
            all_labels.push(Label::new("le", le_str));

            // Register and store the gauge handle
            let g = gauge!(self.name, all_labels);
            gauges.push(g);
        }

        GaugeHistogramHandle { gauges }
    }

    /// Register with no additional labels (just gt/le).
    pub fn register_no_labels(&self) -> GaugeHistogramHandle {
        self.register(&[])
    }
}

// =============================================================================
// CACHED GAUGE HISTOGRAM (for dynamic labels discovered at runtime)
// =============================================================================

/// A gauge histogram with automatic handle caching for dynamic labels.
///
/// Use this when label values (like worker names) are discovered at runtime.
/// Handles are registered on first use and cached for subsequent calls.
///
/// # Example
///
/// ```ignore
/// static BOUNDS: BucketBounds<10> = BucketBounds::new([1, 2, 3, 5, 7, 10, 20, 50, 100, 200]);
/// static HISTOGRAM: GaugeHistogramVec<10> = GaugeHistogramVec::new("smg_worker_dist", &BOUNDS);
///
/// // Create cached wrapper (do this once, store in your router state)
/// let cached = CachedGaugeHistogram::new(&HISTOGRAM);
///
/// // Hot path - first call registers, subsequent calls use cached handle
/// cached.observe("worker-1", &request_counts);
/// cached.observe("worker-2", &request_counts);
/// cached.observe("worker-1", &request_counts);  // Uses cached handle
/// ```
pub struct CachedGaugeHistogram<const N: usize> {
    histogram: &'static GaugeHistogramVec<N>,
    /// Cache of label value -> (handle, counts_buffer)
    cache: DashMap<Arc<str>, (GaugeHistogramHandle, Vec<usize>)>,
    /// Static label key (e.g., "worker", "model")
    label_key: &'static str,
}

impl<const N: usize> CachedGaugeHistogram<N> {
    /// Create a new cached histogram for a single dynamic label.
    ///
    /// # Arguments
    ///
    /// - `histogram`: The static histogram factory
    /// - `label_key`: The label key for the dynamic value (e.g., "worker")
    pub fn new(histogram: &'static GaugeHistogramVec<N>, label_key: &'static str) -> Self {
        Self {
            histogram,
            cache: DashMap::new(),
            label_key,
        }
    }

    /// Get or create a handle for the given label value.
    ///
    /// First call for a label value registers the gauges (allocates).
    /// Subsequent calls return the cached handle (no allocation).
    /// Thread-safe: uses entry API to avoid race conditions.
    pub fn get_or_register(
        &self,
        label_value: &str,
    ) -> dashmap::mapref::one::Ref<'_, Arc<str>, (GaugeHistogramHandle, Vec<usize>)> {
        // Fast path: already cached
        if let Some(entry) = self.cache.get(label_value) {
            return entry;
        }

        // Slow path: use entry API to handle concurrent inserts atomically
        self.cache.entry(Arc::from(label_value)).or_insert_with(|| {
            let handle = self.histogram.register(&[(self.label_key, label_value)]);
            let counts_buf = vec![0usize; self.histogram.bounds.bucket_count()];
            (handle, counts_buf)
        });

        self.cache.get(label_value).unwrap()
    }

    /// Observe a distribution for a label value. **Zero allocation after first call.**
    ///
    /// First call for a new label value registers gauges (allocates).
    /// All subsequent calls are zero-allocation.
    /// Thread-safe: uses entry API to avoid race conditions.
    pub fn observe(&self, label_value: &str, observations: &[u64]) {
        // Fast path: existing entry
        if let Some(mut entry) = self.cache.get_mut(label_value) {
            let (ref handle, ref mut counts_buf) = entry.value_mut();
            self.histogram
                .bounds
                .compute_counts_into(counts_buf, observations);
            handle.set_counts(counts_buf);
            return;
        }

        // Slow path: use entry API to handle concurrent inserts atomically
        let mut entry = self.cache.entry(Arc::from(label_value)).or_insert_with(|| {
            let handle = self.histogram.register(&[(self.label_key, label_value)]);
            let counts_buf = vec![0usize; self.histogram.bounds.bucket_count()];
            (handle, counts_buf)
        });

        let (ref handle, ref mut counts_buf) = entry.value_mut();
        self.histogram
            .bounds
            .compute_counts_into(counts_buf, observations);
        handle.set_counts(counts_buf);
    }

    /// Number of cached label combinations.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Remove a worker and zero out its metrics. **Zero allocation.**
    ///
    /// Call this when a worker is removed from the pool.
    /// Sets all bucket counts to 0 (so Grafana shows it as empty).
    ///
    /// Note: The gauge handles remain in the Prometheus registry (the `metrics`
    /// crate doesn't support unregistering). But memory in our cache is freed.
    pub fn remove(&self, label_value: &str) {
        if let Some((_, (handle, _))) = self.cache.remove(label_value) {
            handle.zero_counts();
        }
    }

    /// Remove workers not in the provided set.
    ///
    /// Call this periodically with your current active workers to clean up stale entries.
    /// Uses `DashMap::retain` for atomic operation without intermediate allocation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let active: HashSet<&str> = workers.iter().map(|w| w.name.as_str()).collect();
    /// cached.retain_only(&active);
    /// ```
    pub fn retain_only<S: std::borrow::Borrow<str> + std::hash::Hash + Eq>(
        &self,
        active_labels: &std::collections::HashSet<S>,
    ) {
        self.cache.retain(|key, (handle, _)| {
            if active_labels.contains(key.as_ref()) {
                true
            } else {
                handle.zero_counts();
                false
            }
        });
    }

    /// Get all currently tracked label values.
    pub fn tracked_labels(&self) -> Vec<Arc<str>> {
        self.cache.iter().map(|e| Arc::clone(e.key())).collect()
    }
}

// =============================================================================
// CONVENIENCE CONSTANTS
// =============================================================================

/// Common bucket bounds for request counts.
pub static REQUEST_COUNT_BOUNDS: BucketBounds<10> =
    BucketBounds::new([1, 2, 3, 5, 7, 10, 20, 50, 100, 200]);

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_bounds_creation() {
        let bounds = BucketBounds::new([10, 30, 60]);
        assert_eq!(bounds.bucket_count(), 4);
        assert_eq!(bounds.bound_count(), 3);
    }

    #[test]
    fn test_bucket_bounds_const_creation() {
        static BOUNDS: BucketBounds<3> = BucketBounds::new([10, 30, 60]);
        assert_eq!(BOUNDS.bucket_count(), 4);
    }

    #[test]
    #[should_panic(expected = "bucket bounds must be strictly ascending")]
    fn test_bucket_bounds_not_ascending_panics() {
        let _ = BucketBounds::new([10, 5, 60]);
    }

    #[test]
    fn test_bucket_index() {
        let bounds = BucketBounds::new([10, 30, 60]);
        assert_eq!(bounds.bucket_index(0), 0);
        assert_eq!(bounds.bucket_index(10), 0);
        assert_eq!(bounds.bucket_index(11), 1);
        assert_eq!(bounds.bucket_index(30), 1);
        assert_eq!(bounds.bucket_index(31), 2);
        assert_eq!(bounds.bucket_index(60), 2);
        assert_eq!(bounds.bucket_index(61), 3);
    }

    #[test]
    fn test_compute_counts() {
        let bounds = BucketBounds::new([10, 30, 60]);
        assert_eq!(
            bounds.compute_counts(&[5, 10, 15, 40, 100]),
            vec![2, 1, 1, 1]
        );
    }

    #[test]
    fn test_compute_counts_into() {
        let bounds = BucketBounds::new([10, 30, 60]);
        let mut counts = [0usize; 4];
        bounds.compute_counts_into(&mut counts, &[5, 10, 15, 40, 100]);
        assert_eq!(counts, [2, 1, 1, 1]);
    }

    #[test]
    fn test_gauge_histogram_vec_creation() {
        static BOUNDS: BucketBounds<3> = BucketBounds::new([10, 30, 60]);
        static HISTOGRAM: GaugeHistogramVec<3> = GaugeHistogramVec::new("test_metric", &BOUNDS);

        assert_eq!(HISTOGRAM.name(), "test_metric");
        assert_eq!(HISTOGRAM.bounds().bucket_count(), 4);
    }

    #[test]
    fn test_gauge_histogram_handle_registration() {
        static BOUNDS: BucketBounds<3> = BucketBounds::new([10, 30, 60]);
        static HISTOGRAM: GaugeHistogramVec<3> = GaugeHistogramVec::new("test_hist", &BOUNDS);

        let handle = HISTOGRAM.register(&[("router", "rr")]);
        assert_eq!(handle.bucket_count(), 4);

        // This should be zero-allocation
        handle.set_counts(&[1, 2, 3, 4]);
    }

    #[test]
    fn test_request_count_bounds() {
        assert_eq!(REQUEST_COUNT_BOUNDS.bucket_count(), 11);
        assert_eq!(REQUEST_COUNT_BOUNDS.bucket_index(1), 0);
        assert_eq!(REQUEST_COUNT_BOUNDS.bucket_index(2), 1);
        assert_eq!(REQUEST_COUNT_BOUNDS.bucket_index(201), 10);
    }

    #[test]
    fn test_cached_histogram() {
        static BOUNDS: BucketBounds<3> = BucketBounds::new([10, 30, 60]);
        static HISTOGRAM: GaugeHistogramVec<3> = GaugeHistogramVec::new("test_cached", &BOUNDS);

        let cached = CachedGaugeHistogram::new(&HISTOGRAM, "worker");

        // First call registers
        cached.observe("worker-1", &[5, 15, 45, 100]);
        assert_eq!(cached.cache_size(), 1);

        // Second call uses cache
        cached.observe("worker-1", &[1, 2, 3]);
        assert_eq!(cached.cache_size(), 1);

        // New worker registers
        cached.observe("worker-2", &[10, 20, 30]);
        assert_eq!(cached.cache_size(), 2);
    }

    #[test]
    fn test_cached_histogram_removal() {
        static BOUNDS: BucketBounds<3> = BucketBounds::new([10, 30, 60]);
        static HISTOGRAM: GaugeHistogramVec<3> =
            GaugeHistogramVec::new("test_cached_remove", &BOUNDS);

        let cached = CachedGaugeHistogram::new(&HISTOGRAM, "worker");

        // Add some workers
        cached.observe("worker-1", &[5, 15]);
        cached.observe("worker-2", &[10, 20]);
        cached.observe("worker-3", &[1, 2]);
        assert_eq!(cached.cache_size(), 3);

        // Remove one
        cached.remove("worker-2");
        assert_eq!(cached.cache_size(), 2);

        // retain_only
        let active: std::collections::HashSet<&str> = ["worker-1"].into_iter().collect();
        cached.retain_only(&active);
        assert_eq!(cached.cache_size(), 1);

        // Check tracked labels
        let labels = cached.tracked_labels();
        assert_eq!(labels.len(), 1);
        assert_eq!(&*labels[0], "worker-1");
    }
}
