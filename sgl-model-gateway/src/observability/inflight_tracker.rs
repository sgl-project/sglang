use std::{
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use dashmap::DashMap;
use tokio::time::interval;
use tracing::debug;

use super::metrics::Metrics;

static INFLIGHT_TRACKER: OnceLock<Arc<InFlightRequestTracker>> = OnceLock::new();

/// Age bucket definitions for in-flight request metrics.
/// Each bucket is (upper_bound_secs, label).
/// The last bucket has no upper bound (infinity).
const AGE_BUCKETS: &[(u64, &str)] = &[
    (30, "0-30s"),
    (60, "30s-1m"),
    (180, "1m-3m"),
    (300, "3m-5m"),
    (600, "5m-10m"),
    (u64::MAX, "10m+"),
];

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

    /// Samples all in-flight requests and records counts per age bucket.
    fn sample_and_record(&self) {
        let now = Instant::now();

        // Initialize bucket counts
        let mut counts = [0usize; AGE_BUCKETS.len()];

        // Count requests in each bucket
        for entry in self.requests.iter() {
            let age_secs = now.duration_since(*entry.value()).as_secs();
            let mut prev_bound = 0u64;
            for (i, &(upper_bound, _)) in AGE_BUCKETS.iter().enumerate() {
                if age_secs >= prev_bound && (upper_bound == u64::MAX || age_secs < upper_bound) {
                    counts[i] += 1;
                    break;
                }
                prev_bound = upper_bound;
            }
        }

        // Set gauge for each bucket
        for (i, &(_, label)) in AGE_BUCKETS.iter().enumerate() {
            Metrics::set_inflight_request_count(label, counts[i]);
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
}

