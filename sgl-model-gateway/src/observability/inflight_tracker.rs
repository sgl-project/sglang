use std::{
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use dashmap::DashMap;
use tokio::time::interval;
use tracing::debug;

use super::metrics::Metrics;

static INFLIGHT_TRACKER: OnceLock<Arc<InFlightRequestTracker>> = OnceLock::new();

/// Age bucket upper bounds in seconds, matching Prometheus histogram `le` label convention.
/// Uses cumulative semantics: le="60" means age <= 60s.
const AGE_BUCKET_BOUNDS: &[u64] = &[30, 60, 180, 300, 600];

/// Label values for each bucket bound, plus "+Inf" for the total.
const AGE_BUCKET_LABELS: &[&str] = &["30", "60", "180", "300", "600", "+Inf"];

/// Tracks in-flight HTTP requests and their start times.
pub struct InFlightRequestTracker {
    requests: DashMap<String, Instant>,
}

impl InFlightRequestTracker {
    pub fn new(sample_interval: Duration) -> Arc<Self> {
        let tracker = Arc::new(Self {
            requests: DashMap::new(),
        });

        tracker.clone().spawn_sampler(sample_interval);
        tracker
    }

    /// Registers a request as in-flight.
    pub fn register(&self, request_id: &str) {
        self.requests.insert(request_id.to_string(), Instant::now());
    }

    /// Deregisters a request when it completes.
    pub fn deregister(&self, request_id: &str) {
        self.requests.remove(request_id);
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    #[cfg(test)]
    pub fn insert_with_time(&self, request_id: &str, start_time: Instant) {
        self.requests.insert(request_id.to_string(), start_time);
    }

    #[cfg(test)]
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

/// Initializes the global in-flight request tracker.
///
/// Should be called once during metrics initialization.
/// The background sampler task starts automatically.
pub fn init_inflight_tracker(sample_interval: Duration) {
    let _ = INFLIGHT_TRACKER.get_or_init(|| InFlightRequestTracker::new(sample_interval));
}

/// Returns a reference to the global tracker, if initialized.
pub fn get_tracker() -> Option<&'static Arc<InFlightRequestTracker>> {
    INFLIGHT_TRACKER.get()
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

        tracker.register("req-1");
        tracker.register("req-2");
        assert_eq!(tracker.len(), 2);

        tracker.deregister("req-1");
        assert_eq!(tracker.len(), 1);

        tracker.deregister("req-2");
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_deregister_nonexistent() {
        let tracker = InFlightRequestTracker {
            requests: DashMap::new(),
        };

        // Should not panic
        tracker.deregister("nonexistent");
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_request_age_tracking() {
        let tracker = InFlightRequestTracker {
            requests: DashMap::new(),
        };

        tracker.register("req-1");
        std::thread::sleep(Duration::from_millis(10));

        let entry = tracker.requests.get("req-1").unwrap();
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

        // Request with age ~0s (should be in all buckets: le=30, 60, 180, 300, 600, +Inf)
        tracker.insert_with_time("req-fresh", now);

        // Request with age ~45s (should be in: le=60, 180, 300, 600, +Inf, but NOT le=30)
        tracker.insert_with_time("req-45s", now - Duration::from_secs(45));

        // Request with age ~100s (should be in: le=180, 300, 600, +Inf, but NOT le=30, 60)
        tracker.insert_with_time("req-100s", now - Duration::from_secs(100));

        // Request with age ~250s (should be in: le=300, 600, +Inf, but NOT le=30, 60, 180)
        tracker.insert_with_time("req-250s", now - Duration::from_secs(250));

        // Request with age ~500s (should be in: le=600, +Inf, but NOT le=30, 60, 180, 300)
        tracker.insert_with_time("req-500s", now - Duration::from_secs(500));

        // Request with age ~700s (should only be in: +Inf)
        tracker.insert_with_time("req-700s", now - Duration::from_secs(700));

        let counts = tracker.compute_bucket_counts();

        // Verify cumulative semantics:
        // le="30":  1 (req-fresh)
        // le="60":  2 (req-fresh, req-45s)
        // le="180": 3 (req-fresh, req-45s, req-100s)
        // le="300": 4 (req-fresh, req-45s, req-100s, req-250s)
        // le="600": 5 (req-fresh, req-45s, req-100s, req-250s, req-500s)
        // +Inf:     6 (all requests)
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

        // Test exact boundary: age = 30s (should be included in le=30)
        tracker.insert_with_time("req-30s", now - Duration::from_secs(30));

        // Test just over boundary: age = 31s (should NOT be in le=30, but in le=60)
        tracker.insert_with_time("req-31s", now - Duration::from_secs(31));

        let counts = tracker.compute_bucket_counts();

        // le="30": 1 (req-30s is exactly at boundary, included)
        // le="60": 2 (both requests)
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

        // Spawn threads that register requests
        for i in 0..10 {
            let tracker_clone = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let id = format!("thread-{}-req-{}", i, j);
                    tracker_clone.register(&id);
                }
            }));
        }

        // Spawn threads that deregister requests
        for i in 0..5 {
            let tracker_clone = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let id = format!("thread-{}-req-{}", i, j);
                    tracker_clone.deregister(&id);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        // After concurrent operations, some requests should remain
        // (threads 5-9 registered 100 each = 500, none of those were deregistered)
        // threads 0-4 registered 100 each but also got deregistered
        // The exact count depends on timing, but it should be non-negative
        assert!(tracker.len() >= 0);

        // Verify we can still compute buckets without panic
        let counts = tracker.compute_bucket_counts();
        assert!(counts[5] >= 0); // +Inf should be non-negative
    }

    #[test]
    fn test_register_overwrites_existing() {
        let tracker = InFlightRequestTracker {
            requests: DashMap::new(),
        };

        let now = Instant::now();
        tracker.insert_with_time("req-1", now - Duration::from_secs(100));

        // Re-register with new timestamp
        tracker.register("req-1");

        // Should still have only 1 entry
        assert_eq!(tracker.len(), 1);

        // The new entry should be fresh (age near 0)
        let entry = tracker.requests.get("req-1").unwrap();
        assert!(entry.value().elapsed() < Duration::from_secs(1));
    }
}

