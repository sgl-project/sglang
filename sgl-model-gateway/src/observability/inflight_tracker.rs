use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, LazyLock, OnceLock,
    },
    time::Instant,
};

use dashmap::DashMap;

use super::metrics::Metrics;
use crate::policies::utils::PeriodicTask;

struct NumericalBuckets {
    upper_bounds: &'static [u64],
    le_labels: Vec<&'static str>,
    gt_labels: Vec<&'static str>,
}

impl NumericalBuckets {
    fn new(upper_bounds: &'static [u64]) -> Self {
        let leak_str = |n: u64| Box::leak(n.to_string().into_boxed_str()) as &'static str;

        let mut le_labels: Vec<&'static str> = upper_bounds.iter().map(|&b| leak_str(b)).collect();
        le_labels.push("+Inf");

        let mut gt_labels: Vec<&'static str> = vec!["0"];
        gt_labels.extend(upper_bounds.iter().map(|&b| leak_str(b)));

        Self {
            upper_bounds,
            le_labels,
            gt_labels,
        }
    }

    fn len(&self) -> usize {
        self.le_labels.len()
    }
}

const AGE_BUCKET_BOUNDS: &[u64] = &[30, 60, 180, 300, 600, 1200, 3600, 7200, 14400, 28800, 86400];
static AGE_BUCKETS: LazyLock<NumericalBuckets> =
    LazyLock::new(|| NumericalBuckets::new(AGE_BUCKET_BOUNDS));

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
        let buckets = &*AGE_BUCKETS;
        let now = Instant::now();
        let inf_idx = buckets.len() - 1;

        let mut counts = vec![0usize; buckets.len()];
        for entry in self.requests.iter() {
            let age_secs = now.duration_since(*entry.value()).as_secs();
            let bucket_idx = buckets
                .upper_bounds
                .iter()
                .position(|&bound| age_secs <= bound)
                .unwrap_or(inf_idx);
            counts[bucket_idx] += 1;
        }

        counts
    }

    fn sample_and_record(&self) {
        let buckets = &*AGE_BUCKETS;
        let counts = self.compute_bucket_counts();
        for ((&gt, &le), &count) in std::iter::zip(
            std::iter::zip(&buckets.gt_labels, &buckets.le_labels),
            &counts,
        ) {
            Metrics::set_inflight_request_age_count(gt, le, count);
        }
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
    fn test_empty_tracker_buckets() {
        let tracker = InFlightRequestTracker::new();
        let counts = tracker.compute_bucket_counts();
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_bucket_counts() {
        let tracker = InFlightRequestTracker::new();
        let now = Instant::now();

        tracker.insert_with_time(1, now);
        tracker.insert_with_time(2, now - Duration::from_secs(45));
        tracker.insert_with_time(3, now - Duration::from_secs(100));
        tracker.insert_with_time(4, now - Duration::from_secs(250));
        tracker.insert_with_time(5, now - Duration::from_secs(500));
        tracker.insert_with_time(6, now - Duration::from_secs(700));

        let counts = tracker.compute_bucket_counts();
        assert_eq!(counts[0], 1, "bucket <=30s");
        assert_eq!(counts[1], 1, "bucket <=60s");
        assert_eq!(counts[2], 1, "bucket <=180s");
        assert_eq!(counts[3], 1, "bucket <=300s");
        assert_eq!(counts[4], 1, "bucket <=600s");
        assert_eq!(counts[5], 1, "bucket <=1200s");
        assert_eq!(*counts.last().unwrap(), 0, "bucket +Inf");
    }

    #[test]
    fn test_bucket_boundary_values() {
        let tracker = InFlightRequestTracker::new();
        let now = Instant::now();

        tracker.insert_with_time(1, now - Duration::from_secs(30));
        tracker.insert_with_time(2, now - Duration::from_secs(31));

        let counts = tracker.compute_bucket_counts();
        assert_eq!(counts[0], 1, "bucket <=30s includes exact boundary");
        assert_eq!(counts[1], 1, "bucket <=60s has one request (31s)");
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

    #[test]
    fn test_age_buckets_labels() {
        let buckets = NumericalBuckets::new(&[10, 30, 60]);

        assert_eq!(buckets.le_labels, vec!["10", "30", "60", "+Inf"]);
        assert_eq!(buckets.gt_labels, vec!["0", "10", "30", "60"]);
        assert_eq!(buckets.len(), 4);
    }

    #[test]
    fn test_age_buckets_global() {
        let buckets = &*AGE_BUCKETS;

        assert_eq!(buckets.le_labels.first(), Some(&"30"));
        assert_eq!(buckets.le_labels.last(), Some(&"+Inf"));
        assert_eq!(buckets.gt_labels.first(), Some(&"0"));
        assert_eq!(buckets.gt_labels.last(), Some(&"86400"));
        assert_eq!(buckets.le_labels.len(), buckets.gt_labels.len());
    }
}
