//! Load-aware worker selection policy
//!
//! This module provides a simple, fast load balancing policy that selects
//! worker(s) based purely on current load metrics.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RoutingContext};
use crate::core::Worker;
use std::sync::Arc;
use tracing::debug;

/// Load-aware policy: Selects worker(s) with minimum load
///
/// This policy provides simple, fast worker selection based purely on current load.
/// It uses `worker.load()` to find the least loaded worker(s).
///
#[derive(Debug, Clone)]
pub struct LoadAwarePolicy {
    name: &'static str,
}

impl LoadAwarePolicy {
    pub fn new() -> Self {
        Self { name: "load_aware" }
    }

    /// Find the index of the worker with minimum load
    pub fn find_min_load_worker(
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
    ) -> Option<usize> {
        if healthy_indices.is_empty() {
            return None;
        }

        let mut min_load = usize::MAX;
        let mut min_idx = healthy_indices[0];

        for &idx in healthy_indices {
            let load = workers[idx].load();
            if load < min_load {
                min_load = load;
                min_idx = idx;
            }
        }

        debug!(
            "LoadAwarePolicy selected worker {} with load {}",
            workers[min_idx].url(),
            min_load
        );

        Some(min_idx)
    }
}

impl Default for LoadAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingPolicy for LoadAwarePolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _context: &RoutingContext,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        Self::find_min_load_worker(workers, &healthy_indices)
    }

    fn select_worker_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        _context: &RoutingContext,
    ) -> Option<(usize, usize)> {
        let prefill_healthy = get_healthy_worker_indices(prefill_workers);
        let decode_healthy = get_healthy_worker_indices(decode_workers);

        let prefill_idx = Self::find_min_load_worker(prefill_workers, &prefill_healthy)?;
        let decode_idx = Self::find_min_load_worker(decode_workers, &decode_healthy)?;

        Some((prefill_idx, decode_idx))
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn create_test_worker(url: &str, priority: u32, cost: f32, load: usize) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url.to_string())
            .worker_type(WorkerType::Regular)
            .label("priority", priority.to_string())
            .label("cost", cost.to_string())
            .build();

        // Set load
        for _ in 0..load {
            worker.increment_load();
        }

        Arc::new(worker) as Arc<dyn Worker>
    }

    #[test]
    fn test_load_aware_policy_selects_min_load() {
        let policy = LoadAwarePolicy::new();
        let workers = vec![
            create_test_worker("http://w1", 50, 1.0, 5),
            create_test_worker("http://w2", 50, 1.0, 2), // Min load
            create_test_worker("http://w3", 50, 1.0, 8),
        ];

        let context = RoutingContext::from_text(None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        assert_eq!(idx, 1); // Should select w2 with load=2
    }

    #[test]
    fn test_load_aware_policy_empty_workers() {
        let policy = LoadAwarePolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];

        let context = RoutingContext::from_text(None);
        let idx = policy.select_worker(&workers, &context);
        assert_eq!(idx, None);
    }

    #[test]
    fn test_load_aware_policy_all_unhealthy() {
        let policy = LoadAwarePolicy::new();
        let workers = vec![
            create_test_worker("http://w1", 50, 1.0, 2),
            create_test_worker("http://w2", 50, 1.0, 3),
        ];

        // Mark all workers as unhealthy
        workers[0].set_healthy(false);
        workers[1].set_healthy(false);

        let context = RoutingContext::from_text(None);
        let idx = policy.select_worker(&workers, &context);
        assert_eq!(idx, None);
    }

    #[test]
    fn test_load_aware_policy_single_healthy_worker() {
        let policy = LoadAwarePolicy::new();
        let workers = vec![
            create_test_worker("http://w1", 50, 1.0, 5),
            create_test_worker("http://w2", 50, 1.0, 2),
            create_test_worker("http://w3", 50, 1.0, 1),
        ];

        // Mark all but one as unhealthy
        workers[0].set_healthy(false);
        workers[2].set_healthy(false);

        let context = RoutingContext::from_text(None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        assert_eq!(idx, 1); // Only w2 is healthy
    }

    #[test]
    fn test_load_aware_policy_zero_load() {
        let policy = LoadAwarePolicy::new();
        let workers = vec![
            create_test_worker("http://w1", 50, 1.0, 5),
            create_test_worker("http://w2", 50, 1.0, 0), // Zero load
            create_test_worker("http://w3", 50, 1.0, 2),
        ];

        let context = RoutingContext::from_text(None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        assert_eq!(idx, 1); // Should select worker with zero load
    }

    #[test]
    fn test_load_aware_policy_tie_breaking() {
        let policy = LoadAwarePolicy::new();
        let workers = vec![
            create_test_worker("http://w1", 50, 1.0, 3),
            create_test_worker("http://w2", 50, 1.0, 3), // Same load
            create_test_worker("http://w3", 50, 1.0, 5),
        ];

        let context = RoutingContext::from_text(None);
        let idx = policy.select_worker(&workers, &context).unwrap();
        // Should select first worker with min load (stable selection)
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_pd_mode_load_aware_policy() {
        let policy = LoadAwarePolicy::new();
        let prefill = vec![
            create_test_worker("http://p1", 50, 1.0, 3),
            create_test_worker("http://p2", 50, 1.0, 1), // Min load
        ];
        let decode = vec![
            create_test_worker("http://d1", 50, 1.0, 5),
            create_test_worker("http://d2", 50, 1.0, 2), // Min load
        ];

        let context = RoutingContext::from_text(None);
        let (p_idx, d_idx) = policy
            .select_worker_pair(&prefill, &decode, &context)
            .unwrap();
        assert_eq!(p_idx, 1); // p2 with load=1
        assert_eq!(d_idx, 1); // d2 with load=2
    }

    #[test]
    fn test_load_aware_policy_pd_mode_empty_prefill() {
        let policy = LoadAwarePolicy::new();
        let prefill: Vec<Arc<dyn Worker>> = vec![];
        let decode = vec![create_test_worker("http://d1", 50, 1.0, 2)];

        let context = RoutingContext::from_text(None);
        let result = policy.select_worker_pair(&prefill, &decode, &context);
        assert_eq!(result, None);
    }

    #[test]
    fn test_load_aware_policy_pd_mode_empty_decode() {
        let policy = LoadAwarePolicy::new();
        let prefill = vec![create_test_worker("http://p1", 50, 1.0, 2)];
        let decode: Vec<Arc<dyn Worker>> = vec![];

        let context = RoutingContext::from_text(None);
        let result = policy.select_worker_pair(&prefill, &decode, &context);
        assert_eq!(result, None);
    }

    #[test]
    fn test_load_aware_policy_load_distribution() {
        let policy = LoadAwarePolicy::new();
        let workers = vec![
            create_test_worker("http://w1", 50, 1.0, 0),
            create_test_worker("http://w2", 50, 1.0, 0),
            create_test_worker("http://w3", 50, 1.0, 0),
        ];

        // Simulate multiple requests and verify load-aware selection
        let context = RoutingContext::from_text(None);
        for _i in 0..10 {
            let idx = policy.select_worker(&workers, &context).unwrap();
            workers[idx].increment_load();

            // After each selection, the selected worker should have the minimum load
            let selected_load = workers[idx].load();
            for (j, w) in workers.iter().enumerate() {
                if j != idx {
                    assert!(w.load() >= selected_load - 1);
                }
            }
        }
    }
}
