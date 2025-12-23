//! Load balancing policies for SGLang router
//!
//! This module provides a unified abstraction for routing policies that work
//! across both regular and prefill-decode (PD) routing modes.

use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, RwLock},
};

use tracing::debug;

use crate::core::Worker;

mod bucket;
mod cache_aware;
mod factory;
mod power_of_two;
mod random;
mod registry;
mod round_robin;
pub mod tree;

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
                return loads
                    .iter()
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
mod dp_load_manager_tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_new_dp_load_manager_instance() {
        let dp_load_manager = DPLoadManager::new();
        let cached = dp_load_manager.dp_cached_loads.read().unwrap();
        assert!(cached.is_empty());
    }

    #[test]
    fn test_update_dp_load() {
        let manager = DPLoadManager::new();
        let mut loads = HashMap::new();

        // insert worker1_load
        let mut worker1_load = HashMap::new();
        worker1_load.insert(0, 2);
        worker1_load.insert(1, 1);
        loads.insert("http://worker1:8080".to_string(), worker1_load);

        // insert worker2.load
        let mut worker2_load = HashMap::new();
        worker2_load.insert(0, 3);
        loads.insert("http://worker2:8080".to_string(), worker2_load);

        // update
        manager.update_dp_loads(&loads);

        // assert
        let cached = manager.dp_cached_loads.read().unwrap();
        assert_eq!(cached.len(), 2);

        let worker2_cache = cached.get("http://worker2:8080").unwrap();
        assert_eq!(worker2_cache.get(&0), Some(&3));
    }

    #[test]
    fn test_get_lowest_dp_load() {
        let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test_api_key2")
            .build();

        let manager = DPLoadManager::new();
        let mut loads = HashMap::new();
        // insert worker1_load
        let mut worker1_load = HashMap::new();
        worker1_load.insert(0, 2);
        worker1_load.insert(1, 1);
        worker1_load.insert(3, 3);
        loads.insert(worker1.url().to_string(), worker1_load);
        manager.update_dp_loads(&loads);

        // Verify that the worker1 with the lowest load is dp_rank = 1
        assert_eq!(manager.get_lowest_dp_load(&worker1), Some(1));
    }

    #[test]
    fn test_load_increment() {
        let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test_api_key2")
            .build();

        let manager = DPLoadManager::new();
        manager.load_increment(&worker2, 0, 5);
        let cached = manager.dp_cached_loads.read().expect("Rwlock read1 failed");
        assert!(cached.get(worker2.url()).is_none());
        drop(cached);

        // insert worker2.load
        let mut worker2_load = HashMap::new();
        worker2_load.insert(0, 2);
        let mut loads = HashMap::new();
        loads.insert(worker2.url().to_string(), worker2_load);
        manager.update_dp_loads(&loads);

        // load increment
        manager.load_increment(&worker2, 0, 5);
        let cached = manager.dp_cached_loads.read().expect("Rwlock read2 failed");
        let worker2_cache = cached
            .get(worker2.url())
            .expect("worker2 not found in cache");
        // 2 + 5 = 7
        assert_eq!(worker2_cache.get(&0), Some(&7));
    }
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
