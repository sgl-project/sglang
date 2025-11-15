//! Load balancing policies for SGLang router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use std::{fmt::Debug, sync::Arc, sync::RwLock};
use std::collections::{HashMap};
use tracing::{debug};

use crate::core::Worker;

mod bucket;
mod cache_aware;
mod factory;
mod power_of_two;
mod random;
mod registry;
mod round_robin;
mod tree;

pub use bucket::BucketPolicy;
pub use cache_aware::CacheAwarePolicy;
pub use factory::PolicyFactory;
pub use power_of_two::PowerOfTwoPolicy;
pub use random::RandomPolicy;
pub use registry::PolicyRegistry;
pub use round_robin::RoundRobinPolicy;

/// Core trait for load balancing policies
///
/// This trait provides a unified interface for implementing routing algorithms
/// that can work with both regular single-worker selection and PD dual-worker selection.
pub trait LoadBalancingPolicy: Send + Sync + Debug {
    /// Select a single worker from the available workers
    ///
    /// This is used for regular routing mode where requests go to a single worker.
    /// Now uses Arc<dyn Worker> for better performance and to avoid unnecessary cloning.
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<usize>;

    /// Select a pair of workers (prefill and decode) for PD routing
    ///
    /// Returns indices of (prefill_worker, decode_worker) from their respective arrays.
    /// Default implementation uses select_worker for each array independently.
    fn select_worker_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<(usize, usize)> {
        // Default implementation: independently select from each pool
        let prefill_idx = self.select_worker(prefill_workers, request_text)?;
        let decode_idx = self.select_worker(decode_workers, request_text)?;
        Some((prefill_idx, decode_idx))
    }

    /// Update policy state after request completion
    ///
    /// This is called when a request completes (successfully or not) to allow
    /// policies to update their internal state.
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {
        // Default: no-op for stateless policies
    }

    /// Get policy name for metrics and debugging
    fn name(&self) -> &'static str;

    /// Check if this policy needs request text for routing decisions
    fn needs_request_text(&self) -> bool {
        false // Default: most policies don't need request text
    }

    /// Update worker load information
    ///
    /// This is called periodically with current load information for load-aware policies.
    fn update_loads(&self, _loads: &HashMap<String, isize>) {
        // Default: no-op for policies that don't use load information
    }

    fn update_dp_loads(&self, _loads: &HashMap<String, HashMap<isize, isize>>) {
        // Default: no-op for policies that don't use load information
    }

    fn get_lowest_dp_load(&self, _worker: &dyn Worker) -> Option<isize> {
        None
    }

    fn load_increment(&self, _worker: &dyn Worker, _dp_rank: isize, _tokens: isize) {
        // Default
    }

    /// Reset any internal state
    ///
    /// This is useful for policies that maintain state (e.g., round-robin counters).
    fn reset(&self) {
        // Default: no-op for stateless policies
    }

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Configuration for cache-aware policy
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 10000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub bucket_adjust_interval_secs: usize,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            bucket_adjust_interval_secs: 5,
        }
    }
}

/// Configuration for cache-aware policy
#[derive(Debug, Default)]
pub struct DPLoadManager {
    dp_cached_loads: RwLock<HashMap<String, HashMap<isize, isize>>>,
}

impl DPLoadManager {
    pub fn new() -> Self {
        Self {
            dp_cached_loads: RwLock::new(HashMap::new()),
        }
    }

    pub fn update_dp_loads(&self, loads: &HashMap<String, HashMap<isize, isize>>) {
        debug!("RoundRobinPolicy update_dp_loads map:{:?}", loads);
        if let Ok(mut cached) = self.dp_cached_loads.write() {
            *cached = loads.clone();
        }
    }

    pub fn get_lowest_dp_load(&self, worker: &dyn Worker) -> Option<isize> {
        if let Ok(cached_loads) = self.dp_cached_loads.read() {
            if let Some(loads) = cached_loads.get(worker.url()) {
                return loads.iter()
                    .min_by_key(|&(_, load)| load)
                    .map(|(&rand_id, _)| rand_id);
            }
        }
        None
    }

    pub fn load_increment(&self, worker: &dyn Worker, dp_rank: isize, increment: isize) {
        // Add an increment to the load of dp group,
        // to prevent all request from being scheduled to the same DP group during the interval between two load reports.
        if let Ok(mut cached_loads) = self.dp_cached_loads.write() {
            if let Some(loads) = cached_loads.get_mut(worker.url()) {
                if let Some(dp_load) = loads.get_mut(&dp_rank) {
                    *dp_load += increment;
                }
            }
        }
    }
}

/// Helper function to filter healthy workers and return their indices
pub(crate) fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_healthy() && w.circuit_breaker().can_execute())
        .map(|(idx, _)| idx)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_get_healthy_worker_indices() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key2")
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .build(),
            ),
        ];

        // All healthy initially
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one unhealthy
        workers[1].set_healthy(false);
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }
}
