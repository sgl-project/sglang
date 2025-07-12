//! Random load balancing policy

use super::{get_healthy_worker_indices, LoadBalancingPolicy};
use crate::core::Worker;
use rand::Rng;

/// Random selection policy
///
/// Selects workers randomly with uniform distribution among healthy workers.
#[derive(Debug, Default)]
pub struct RandomPolicy;

impl RandomPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl LoadBalancingPolicy for RandomPolicy {
    fn select_worker(
        &self,
        workers: &[Box<dyn Worker>],
        _request_text: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();
        let random_idx = rng.gen_range(0..healthy_indices.len());
        Some(healthy_indices[random_idx])
    }

    fn name(&self) -> &'static str {
        "random"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use std::collections::HashMap;

    #[test]
    fn test_random_selection() {
        let policy = RandomPolicy::new();
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

        // Test multiple selections to ensure randomness
        let mut counts = HashMap::new();
        for _ in 0..100 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        // All workers should be selected at least once
        assert_eq!(counts.len(), 3);
        assert!(counts.values().all(|&count| count > 0));
    }

    #[test]
    fn test_random_with_unhealthy_workers() {
        let policy = RandomPolicy::new();
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

        // Mark first worker as unhealthy
        workers[0].set_healthy(false);

        // Should always select the healthy worker (index 1)
        for _ in 0..10 {
            assert_eq!(policy.select_worker(&workers, None), Some(1));
        }
    }

    #[test]
    fn test_random_no_healthy_workers() {
        let policy = RandomPolicy::new();
        let workers: Vec<Box<dyn Worker>> = vec![Box::new(BasicWorker::new(
            "http://w1:8000".to_string(),
            WorkerType::Regular,
        ))];

        workers[0].set_healthy(false);
        assert_eq!(policy.select_worker(&workers, None), None);
    }
}
