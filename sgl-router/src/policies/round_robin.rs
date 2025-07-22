//! Round-robin load balancing policy

use super::{get_healthy_worker_indices, LoadBalancingPolicy};
use crate::core::Worker;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Round-robin selection policy
///
/// Selects workers in sequential order, cycling through all healthy workers.
#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl LoadBalancingPolicy for RoundRobinPolicy {
    fn select_worker(
        &self,
        workers: &[Box<dyn Worker>],
        _request_text: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Get and increment counter atomically
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let selected_idx = count % healthy_indices.len();

        Some(healthy_indices[selected_idx])
    }

    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_round_robin_selection() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Box<dyn Worker>> = vec![
            Box::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://w3:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Should select workers in order: 0, 1, 2, 0, 1, 2, ...
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(1));
        assert_eq!(policy.select_worker(&workers, None), Some(2));
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(1));
    }

    #[test]
    fn test_round_robin_with_unhealthy_workers() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Box<dyn Worker>> = vec![
            Box::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://w3:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Mark middle worker as unhealthy
        workers[1].set_healthy(false);

        // Should skip unhealthy worker: 0, 2, 0, 2, ...
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(2));
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(2));
    }

    #[test]
    fn test_round_robin_reset() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Box<dyn Worker>> = vec![
            Box::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Advance the counter
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(1));

        // Reset should start from beginning
        policy.reset();
        assert_eq!(policy.select_worker(&workers, None), Some(0));
    }
}
