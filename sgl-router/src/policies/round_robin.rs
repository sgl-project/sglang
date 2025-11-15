//! Round-robin load balancing policy

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::collections::{HashMap};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, DPLoadManager};
use crate::{core::Worker, metrics::RouterMetrics};

/// Round-robin selection policy
///
/// Selects workers in sequential order, cycling through all healthy workers.
#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
    dp_load_manager: DPLoadManager,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
            dp_load_manager: DPLoadManager::new(),
        }
    }
}

impl LoadBalancingPolicy for RoundRobinPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Get and increment counter atomically
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let selected_idx = count % healthy_indices.len();
        let worker = workers[healthy_indices[selected_idx]].url();

        RouterMetrics::record_processed_request(worker);
        RouterMetrics::record_policy_decision(self.name(), worker);
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

    fn update_dp_loads(&self, loads: &HashMap<String, HashMap<isize, isize>>) {
        return self.dp_load_manager.update_dp_loads(loads);
    }

    fn get_lowest_dp_load(&self, worker: &dyn Worker) -> Option<isize> {
        return self.dp_load_manager.get_lowest_dp_load(worker);
    }

    fn load_increment(&self, worker: &dyn Worker, dp_rank: isize, tokens: isize) {
        return self.dp_load_manager.load_increment(worker, dp_rank, tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_round_robin_selection() {
        let policy = RoundRobinPolicy::new();
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

        // Advance the counter
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(1));

        // Reset should start from beginning
        policy.reset();
        assert_eq!(policy.select_worker(&workers, None), Some(0));
    }
}
