use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, LazyLock, OnceLock,
    },
    time::Instant,
};

use dashmap::DashMap;

use super::gauge_histogram::{BucketBounds, GaugeHistogramHandle, GaugeHistogramVec};
use crate::policies::utils::PeriodicTask;

static INFLIGHT_AGE_BOUNDS: BucketBounds<11> =
    BucketBounds::new([30, 60, 180, 300, 600, 1200, 3600, 7200, 14400, 28800, 86400]);
static INFLIGHT_AGE_HISTOGRAM: GaugeHistogramVec<11> =
    GaugeHistogramVec::new("smg_http_inflight_request_age_count", &INFLIGHT_AGE_BOUNDS);
static INFLIGHT_AGE_HANDLE: LazyLock<GaugeHistogramHandle> =
    LazyLock::new(|| INFLIGHT_AGE_HISTOGRAM.register_no_labels());

pub struct InFlightRequestTracker {
    requests: DashMap<u64, Instant>,
    next_id: AtomicU64,
    sampler: OnceLock<PeriodicTask>,
}

impl InFlightRequestTracker {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            requests: DashMap::new(),
            next_id: AtomicU64::new(0),
            sampler: OnceLock::new(),
        })
    }

    pub fn start_sampler(self: &Arc<Self>, interval_secs: u64) {
        let tracker = self.clone();
        let task = PeriodicTask::spawn(interval_secs, "InFlightRequestSampler", move || {
            tracker.sample_and_record();
        });
        self.sampler.set(task).unwrap();
    }

    pub fn track(self: &Arc<Self>) -> InFlightGuard {
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.requests.insert(request_id, Instant::now());
        InFlightGuard {
            tracker: self.clone(),
            request_id,
        }
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn compute_bucket_counts(&self) -> Vec<usize> {
        let ages = self.collect_ages();
        INFLIGHT_AGE_BOUNDS.compute_counts(&ages)
    }

    fn collect_ages(&self) -> Vec<u64> {
        let now = Instant::now();
        self.requests
            .iter()
            .map(|entry| now.duration_since(*entry.value()).as_secs())
            .collect()
    }

    fn sample_and_record(&self) {
        let counts = self.compute_bucket_counts();
        INFLIGHT_AGE_HANDLE.set_counts(&counts);
    }
}

pub struct InFlightGuard {
    tracker: Arc<InFlightRequestTracker>,
    request_id: u64,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.tracker.requests.remove(&self.request_id);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    impl InFlightRequestTracker {
        fn insert_with_time(&self, request_id: u64, start_time: Instant) {
            self.requests.insert(request_id, start_time);
        }
    }

    #[test]
    fn test_track_and_drop() {
        let tracker = InFlightRequestTracker::new();
        {
            let _guard1 = tracker.track();
            let _guard2 = tracker.track();
            assert_eq!(tracker.len(), 2);
        }
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_guard_auto_deregister() {
        let tracker = InFlightRequestTracker::new();
        let guard = tracker.track();
        assert_eq!(tracker.len(), 1);
        drop(guard);
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_request_age_tracking() {
        let tracker = InFlightRequestTracker::new();
        let _guard = tracker.track();
        std::thread::sleep(Duration::from_millis(100));

        let entry = tracker.requests.iter().next().unwrap();
        let age = entry.value().elapsed();
        assert!(age >= Duration::from_millis(100));
    }

    #[test]
    fn test_collect_ages_empty() {
        let tracker = InFlightRequestTracker::new();
        let ages = tracker.collect_ages();
        assert!(ages.is_empty());
    }

    #[test]
    fn test_collect_ages() {
        let tracker = InFlightRequestTracker::new();
        let now = Instant::now();

        tracker.insert_with_time(1, now);
        tracker.insert_with_time(2, now - Duration::from_secs(45));
        tracker.insert_with_time(3, now - Duration::from_secs(100));

        let ages = tracker.collect_ages();
        assert_eq!(ages.len(), 3);
        // Ages should be approximately 0, 45, 100 (order may vary due to DashMap)
        let mut sorted_ages = ages.clone();
        sorted_ages.sort();
        assert!(sorted_ages[0] <= 1); // ~0s
        assert!((44..=46).contains(&sorted_ages[1])); // ~45s
        assert!((99..=101).contains(&sorted_ages[2])); // ~100s
    }

    #[test]
    fn test_concurrent_tracking() {
        use std::thread;

        let tracker = InFlightRequestTracker::new();
        let mut handles = vec![];

        for _ in 0..10 {
            let t = tracker.clone();
            handles.push(thread::spawn(move || {
                (0..100).map(|_| t.track()).collect::<Vec<_>>()
            }));
        }

        let all_guards: Vec<_> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        assert_eq!(tracker.len(), 1000);
        drop(all_guards);
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_unique_ids() {
        let tracker = InFlightRequestTracker::new();
        let g1 = tracker.track();
        let g2 = tracker.track();
        let g3 = tracker.track();

        assert_ne!(g1.request_id, g2.request_id);
        assert_ne!(g2.request_id, g3.request_id);
        assert_eq!(tracker.len(), 3);
    }
}
