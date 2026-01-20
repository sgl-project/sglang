//! Round-robin load balancing policy

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use async_trait::async_trait;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;

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

#[async_trait]
impl LoadBalancingPolicy for RoundRobinPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
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
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_round_robin_selection() {
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

        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(2));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
    }

    #[tokio::test]
    async fn test_round_robin_with_unhealthy_workers() {
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

        workers[1].set_healthy(false);

        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(2));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(2));
    }

    #[tokio::test]
    async fn test_round_robin_reset() {
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

        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));

        policy.reset();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
    }
}
