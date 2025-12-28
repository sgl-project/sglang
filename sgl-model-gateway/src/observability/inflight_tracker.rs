use std::{
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use dashmap::DashMap;
use tokio::time::interval;
use tracing::debug;

use super::metrics::Metrics;

static INFLIGHT_TRACKER: OnceLock<Arc<InFlightRequestTracker>> = OnceLock::new();

/// Tracks in-flight HTTP requests and their start times.
pub struct InFlightRequestTracker {
    /// Maps request_id -> request start time
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

    /// Samples all in-flight requests and records their ages to the histogram.
    fn sample_and_record(&self) {
        let now = Instant::now();
        for entry in self.requests.iter() {
            let age = now.duration_since(*entry.value());
            Metrics::record_inflight_request_age(age);
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

