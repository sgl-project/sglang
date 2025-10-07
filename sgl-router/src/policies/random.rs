//! Random load balancing policy

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RoutingContext};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use rand::Rng;
use std::sync::Arc;

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
        workers: &[Arc<dyn Worker>],
        _context: &RoutingContext,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        let mut rng = rand::rng();
        let random_idx = rng.random_range(0..healthy_indices.len());
        let worker = workers[healthy_indices[random_idx]].url();

        RouterMetrics::record_processed_request(worker);
        RouterMetrics::record_policy_decision(self.name(), worker);
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
    use crate::core::{BasicWorkerBuilder, WorkerType};
    use std::collections::HashMap;

    #[test]
    fn test_random_selection() {
        let policy = RandomPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let mut counts = HashMap::new();
        let context = super::RoutingContext::from_text(None);
        for _ in 0..100 {
            if let Some(idx) = policy.select_worker(&workers, &context) {
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
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        // Mark first worker as unhealthy
        workers[0].set_healthy(false);

        // Should always select the healthy worker (index 1)
        let context = super::RoutingContext::from_text(None);
        for _ in 0..10 {
            assert_eq!(policy.select_worker(&workers, &context), Some(1));
        }
    }

    #[test]
    fn test_random_no_healthy_workers() {
        let policy = RandomPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        workers[0].set_healthy(false);
        let context = super::RoutingContext::from_text(None);
        assert_eq!(policy.select_worker(&workers, &context), None);
    }
}
