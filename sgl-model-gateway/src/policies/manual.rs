//! Manual routing policy based on routing_id

use std::sync::Arc;

use rand::Rng;

use super::{get_healthy_worker_indices, LoadBalancingPolicy};
use crate::core::Worker;

/// Manual routing policy
///
/// Routes requests based on routing_id field using consistent hashing.
/// Requests with the same routing_id are always routed to the same worker.
/// Falls back to random selection when routing_id is not provided.
#[derive(Debug, Default)]
pub struct ManualPolicy;

impl ManualPolicy {
    pub fn new() -> Self {
        Self
    }

    /// Compute hash for routing_id using a simple hash function
    fn compute_hash(routing_id: &str) -> u64 {
        routing_id
            .bytes()
            .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64))
    }
}

impl LoadBalancingPolicy for ManualPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        routing_id: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Use routing_id for consistent routing
        if let Some(routing_id) = routing_id {
            if !routing_id.is_empty() {
                let hash = Self::compute_hash(routing_id);
                let idx = hash as usize % healthy_indices.len();
                return Some(healthy_indices[idx]);
            }
        }

        // Fallback to random selection when routing_id is not provided
        let mut rng = rand::rng();
        let random_idx = rng.random_range(0..healthy_indices.len());
        Some(healthy_indices[random_idx])
    }

    fn name(&self) -> &'static str {
        "manual"
    }

    fn needs_routing_id(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_manual_consistent_routing() {
        let policy = ManualPolicy::new();
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

        // Same routing_id should always route to the same worker
        let routing_id = "user-123";
        let first_idx = policy
            .select_worker(&workers, None, Some(routing_id))
            .unwrap();

        for _ in 0..10 {
            let idx = policy
                .select_worker(&workers, None, Some(routing_id))
                .unwrap();
            assert_eq!(
                idx, first_idx,
                "Same routing_id should route to same worker"
            );
        }
    }

    #[test]
    fn test_manual_different_routing_ids() {
        let policy = ManualPolicy::new();
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

        // Different routing_ids should distribute across workers
        let mut distribution = HashMap::new();
        for i in 0..100 {
            let routing_id = format!("user-{}", i);
            let idx = policy
                .select_worker(&workers, None, Some(&routing_id))
                .unwrap();
            *distribution.entry(idx).or_insert(0) += 1;
        }

        // Should have at least some distribution (not all to one worker)
        assert!(
            distribution.len() > 1,
            "Should distribute across multiple workers"
        );
    }

    #[test]
    fn test_manual_fallback_random() {
        let policy = ManualPolicy::new();
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

        // Without routing_id, should use random selection
        let mut counts = HashMap::new();
        for _ in 0..100 {
            if let Some(idx) = policy.select_worker(&workers, None, None) {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        // Both workers should be selected at least once
        assert_eq!(counts.len(), 2, "Random fallback should use all workers");
    }

    #[test]
    fn test_manual_with_unhealthy_workers() {
        let policy = ManualPolicy::new();
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

        // Should only select the healthy worker
        for _ in 0..10 {
            let idx = policy
                .select_worker(&workers, None, Some("test-routing-id"))
                .unwrap();
            assert_eq!(idx, 1, "Should only select healthy worker");
        }
    }

    #[test]
    fn test_manual_no_healthy_workers() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        workers[0].set_healthy(false);
        assert_eq!(policy.select_worker(&workers, None, Some("test")), None);
    }

    #[test]
    fn test_manual_empty_routing_id() {
        let policy = ManualPolicy::new();
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

        // Empty routing_id should fall back to random
        let mut counts = HashMap::new();
        for _ in 0..100 {
            if let Some(idx) = policy.select_worker(&workers, None, Some("")) {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        assert_eq!(
            counts.len(),
            2,
            "Empty routing_id should use random fallback"
        );
    }
}
