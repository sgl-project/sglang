use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::Instant,
};

use dashmap::DashMap;

use super::metrics::Metrics;
use crate::policies::utils::PeriodicTask;

const AGE_BUCKET_BOUNDS: &[u64] = &[30, 60, 180, 300, 600, 1200, 3600, 7200, 14400, 28800, 86400];
const AGE_BUCKET_LABELS: &[&str] = &[
    "30", "60", "180", "300", "600", "1200", "3600", "7200", "14400", "28800", "86400", "+Inf",
];

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

    pub fn compute_bucket_counts(&self) -> [usize; AGE_BUCKET_LABELS.len()] {
        let now = Instant::now();
        let inf_idx = AGE_BUCKET_LABELS.len() - 1;

        let instants: Vec<Instant> = self.requests.iter().map(|entry| *entry.value()).collect();

        let mut non_cumulative_counts = [0usize; AGE_BUCKET_LABELS.len()];
        for inst in instants {
            let age_secs = now.duration_since(inst).as_secs();
            let bucket_idx = AGE_BUCKET_BOUNDS
                .iter()
                .position(|&bound| age_secs <= bound)
                .unwrap_or(inf_idx);
            non_cumulative_counts[bucket_idx] += 1;
        }

        let mut counts = [0usize; AGE_BUCKET_LABELS.len()];
        let mut cumulative = 0;
        for i in 0..counts.len() {
            cumulative += non_cumulative_counts[i];
            counts[i] = cumulative;
        }

        counts
    }

    fn sample_and_record(&self) {
        let counts = self.compute_bucket_counts();
        for (i, &label) in AGE_BUCKET_LABELS.iter().enumerate() {
            Metrics::set_inflight_request_age_count(label, counts[i]);
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
    fn test_cumulative_bucket_counts() {
        let tracker = InFlightRequestTracker::new();
        let now = Instant::now();

        tracker.insert_with_time(1, now);
        tracker.insert_with_time(2, now - Duration::from_secs(45));
        tracker.insert_with_time(3, now - Duration::from_secs(100));
        tracker.insert_with_time(4, now - Duration::from_secs(250));
        tracker.insert_with_time(5, now - Duration::from_secs(500));
        tracker.insert_with_time(6, now - Duration::from_secs(700));

        let counts = tracker.compute_bucket_counts();
        assert_eq!(counts[0], 1, "bucket 0");
        assert_eq!(counts[1], 2, "bucket 1");
        assert_eq!(counts[2], 3, "bucket 2");
        assert_eq!(counts[3], 4, "bucket 3");
        assert_eq!(counts[4], 5, "bucket 4");
        assert_eq!(counts[5], 6, "bucket 5");
        assert_eq!(*counts.last().unwrap(), 6, "bucket +Inf");
    }

    #[test]
    fn test_bucket_boundary_values() {
        let tracker = InFlightRequestTracker::new();
        let now = Instant::now();

        tracker.insert_with_time(1, now - Duration::from_secs(30));
        tracker.insert_with_time(2, now - Duration::from_secs(31));

        let counts = tracker.compute_bucket_counts();
        assert_eq!(counts[0], 1, "bucket 0 includes exact boundary");
        assert_eq!(counts[1], 2, "bucket 1 includes both");
        assert_eq!(*counts.last().unwrap(), 2, "bucket +Inf includes all");
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
