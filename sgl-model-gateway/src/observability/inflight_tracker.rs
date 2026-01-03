use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use tokio::time::interval;
use tracing::debug;

use super::metrics::Metrics;

const AGE_BUCKET_BOUNDS: &[u64] = &[30, 60, 180, 300, 600];
const AGE_BUCKET_LABELS: &[&str] = &["30", "60", "180", "300", "600", "+Inf"];

pub struct InFlightRequestTracker {
    requests: DashMap<u64, Instant>,
    next_id: AtomicU64,
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

impl InFlightRequestTracker {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            requests: DashMap::new(),
            next_id: AtomicU64::new(0),
        })
    }

    pub fn new_with_sampler(sample_interval: Duration) -> Arc<Self> {
        let tracker = Self::new();
        tracker.clone().spawn_sampler(sample_interval);
        tracker
    }

    pub fn start_sampler(self: &Arc<Self>, sample_interval: Duration) {
        self.clone().spawn_sampler(sample_interval);
    }

    pub fn track(self: &Arc<Self>) -> InFlightGuard {
        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.requests.insert(request_id, Instant::now());
        InFlightGuard {
            tracker: self.clone(),
            request_id,
        }
    }

    #[doc(hidden)]
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    #[doc(hidden)]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    #[doc(hidden)]
    pub fn insert_with_time(&self, request_id: u64, start_time: Instant) {
        self.requests.insert(request_id, start_time);
    }

    #[doc(hidden)]
    pub fn compute_bucket_counts(&self) -> [usize; 6] {
        let now = Instant::now();
        let mut cumulative_counts = [0usize; AGE_BUCKET_LABELS.len()];
        let total_buckets = AGE_BUCKET_LABELS.len();

        for entry in self.requests.iter() {
            let age_secs = now.duration_since(*entry.value()).as_secs();

            for (i, &bound) in AGE_BUCKET_BOUNDS.iter().enumerate() {
                if age_secs <= bound {
                    cumulative_counts[i] += 1;
                }
            }
            cumulative_counts[total_buckets - 1] += 1;
        }

        cumulative_counts
    }

    /// Samples all in-flight requests and records cumulative counts per age bucket.
    /// Uses histogram-style cumulative semantics: le="60" = count of requests with age <= 60s.
    fn sample_and_record(&self) {
        let now = Instant::now();

        // Initialize cumulative bucket counts (one extra for +Inf)
        let mut cumulative_counts = [0usize; AGE_BUCKET_LABELS.len()];
        let total_buckets = AGE_BUCKET_LABELS.len();

        // Count requests into cumulative buckets
        for entry in self.requests.iter() {
            let age_secs = now.duration_since(*entry.value()).as_secs();

            // Increment all buckets where age <= bound (cumulative)
            for (i, &bound) in AGE_BUCKET_BOUNDS.iter().enumerate() {
                if age_secs <= bound {
                    cumulative_counts[i] += 1;
                }
            }
            // Always increment +Inf bucket
            cumulative_counts[total_buckets - 1] += 1;
        }

        // Set gauge for each bucket with le label
        for (i, &label) in AGE_BUCKET_LABELS.iter().enumerate() {
            Metrics::set_inflight_request_count(label, cumulative_counts[i]);
        }
    }

    /// Spawns the background sampler task.
    fn spawn_sampler(self: Arc<Self>, sample_interval: Duration) {
        tokio::spawn(async move {
            let mut ticker = interval(sample_interval);
            debug!(
                "InFlightRequestTracker sampler started with {}ms interval",
                sample_interval.as_millis()
            );

            loop {
                ticker.tick().await;
                self.sample_and_record();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_register_deregister() {
        let tracker = InFlightRequestTracker {
            requests: DashMap::new(),
        };

        tracker.register(1);
        tracker.register(2);
        assert_eq!(tracker.len(), 2);

        tracker.deregister(1);
        assert_eq!(tracker.len(), 1);

        tracker.deregister(2);
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_deregister_nonexistent() {
        let tracker = InFlightRequestTracker {
            requests: DashMap::new(),
        };

        tracker.deregister(999);
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_request_age_tracking() {
        let tracker = InFlightRequestTracker {
            requests: DashMap::new(),
        };

        tracker.register(1);
        std::thread::sleep(Duration::from_millis(10));

        let entry = tracker.requests.get(&1).unwrap();
        let age = entry.value().elapsed();
        assert!(age >= Duration::from_millis(10));
    }

    #[test]
    fn test_sample_empty_tracker() {
        let tracker = InFlightRequestTracker {
            requests: DashMap::new(),
        };

        let counts = tracker.compute_bucket_counts();
        assert_eq!(counts, [0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_sample_and_record_cumulative_buckets() {
        let tracker = InFlightRequestTracker {
            requests: DashMap::new(),
        };

        let now = Instant::now();
        tracker.insert_with_time(1, now);
        tracker.insert_with_time(2, now - Duration::from_secs(45));
        tracker.insert_with_time(3, now - Duration::from_secs(100));
        tracker.insert_with_time(4, now - Duration::from_secs(250));
        tracker.insert_with_time(5, now - Duration::from_secs(500));
        tracker.insert_with_time(6, now - Duration::from_secs(700));

        let counts = tracker.compute_bucket_counts();
        assert_eq!(counts[0], 1, "le=30 bucket");
        assert_eq!(counts[1], 2, "le=60 bucket");
        assert_eq!(counts[2], 3, "le=180 bucket");
        assert_eq!(counts[3], 4, "le=300 bucket");
        assert_eq!(counts[4], 5, "le=600 bucket");
        assert_eq!(counts[5], 6, "+Inf bucket");
    }

    #[test]
    fn test_cumulative_bucket_boundary_values() {
        let tracker = InFlightRequestTracker {
            requests: DashMap::new(),
        };

        let now = Instant::now();
        tracker.insert_with_time(1, now - Duration::from_secs(30));
        tracker.insert_with_time(2, now - Duration::from_secs(31));

        let counts = tracker.compute_bucket_counts();
        assert_eq!(counts[0], 1, "le=30 should include exact boundary");
        assert_eq!(counts[1], 2, "le=60 should include both");
        assert_eq!(counts[5], 2, "+Inf should include all");
    }

    #[test]
    fn test_concurrent_register_deregister() {
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(InFlightRequestTracker {
            requests: DashMap::new(),
        });

        let mut handles = vec![];

        for i in 0..10u64 {
            let tracker_clone = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for j in 0..100u64 {
                    tracker_clone.register(i * 1000 + j);
                }
            }));
        }

        for i in 0..5u64 {
            let tracker_clone = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for j in 0..100u64 {
                    tracker_clone.deregister(i * 1000 + j);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        let _ = tracker.len();
        let counts = tracker.compute_bucket_counts();
        let _ = counts[5];
    }

    #[test]
    fn test_register_overwrites_existing() {
        let tracker = InFlightRequestTracker {
            requests: DashMap::new(),
        };

        let now = Instant::now();
        tracker.insert_with_time(1, now - Duration::from_secs(100));
        tracker.register(1);

        assert_eq!(tracker.len(), 1);
        let entry = tracker.requests.get(&1).unwrap();
        assert!(entry.value().elapsed() < Duration::from_secs(1));
    }
}

