/*
    Cache-Aware Load Balancing Router

    This router combines two strategies to optimize both cache utilization and request distribution:

    1. Cache-Aware Routing (Approximate Tree)
    2. Load Balancing (Shortest Queue with Balance Thresholds)

    The router dynamically switches between these strategies based on load conditions:
    - Uses load balancing when the system is imbalanced
    - Uses cache-aware routing when the system is balanced

    A system is considered imbalanced if both conditions are met:
    1. (max - min) > abs_threshold
    2. max > rel_threshold * min

    Strategy Details:

    1. Cache-Aware Routing (Approximate Tree)
    -------------------------------------------
    This strategy maintains an approximate radix tree for each worker based on request history,
    eliminating the need for direct cache state queries. The tree stores raw text characters
    instead of token IDs to avoid tokenization overhead.

    Process:
    a. For each request, find the worker with the highest prefix match
    b. If match rate > cache_threshold:
    Route to the worker with highest match (likely has relevant data cached)
    c. If match rate ≤ cache_threshold:
    Route to the worker with smallest tree size (most available cache capacity)
    d. Background maintenance:
    Periodically evict least recently used leaf nodes to prevent memory overflow

    2. Load Balancing (Shortest Queue)
    -------------------------------------------
    This strategy tracks pending request counts per worker and routes new requests
    to the least busy worker when the system is detected to be imbalanced.

    Configuration Parameters:
    ------------------------
    1. cache_threshold: (float, 0.0 to 1.0)
    Minimum prefix match ratio to use highest-match routing.
    Below this threshold, routes to worker with most available cache space.

    2. balance_abs_threshold: (integer)
    Absolute difference threshold for load imbalance detection.
    System is potentially imbalanced if (max_load - min_load) > abs_threshold

    3. balance_rel_threshold: (float)
    Relative ratio threshold for load imbalance detection.
    System is potentially imbalanced if max_load > min_load * rel_threshold
    Used in conjunction with abs_threshold to determine final imbalance state.

    4. eviction_interval_secs: (integer)
    Interval between LRU eviction cycles for the approximate trees.

    5. max_tree_size: (integer)
    Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
    during the next eviction cycle.
*/

use super::{get_healthy_worker_indices, CacheAwareConfig, LoadBalancingPolicy};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::tree::Tree;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tracing::{debug, info};

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    tree: Arc<Mutex<Tree>>,
    eviction_handle: Option<thread::JoinHandle<()>>,
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let tree = Arc::new(Mutex::new(Tree::new()));

        // Start background eviction thread if configured
        let eviction_handle = if config.eviction_interval_secs > 0 {
            let tree_clone = Arc::clone(&tree);
            let max_tree_size = config.max_tree_size;
            let interval = config.eviction_interval_secs;

            Some(thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(interval));

                if let Ok(tree_guard) = tree_clone.lock() {
                    tree_guard.evict_tenant_by_size(max_tree_size);
                    debug!("Cache eviction completed, max_size: {}", max_tree_size);
                }
            }))
        } else {
            None
        };

        Self {
            config,
            tree,
            eviction_handle,
        }
    }

    /// Initialize the tree with worker URLs
    pub fn init_workers(&self, workers: &[Box<dyn Worker>]) {
        if let Ok(tree) = self.tree.lock() {
            for worker in workers {
                tree.insert("", worker.url());
            }
        }
    }

    /// Remove a worker from the tree
    pub fn remove_worker(&self, url: &str) {
        if let Ok(tree) = self.tree.lock() {
            tree.remove_tenant(url);
        }
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        if let Ok(tree) = self.tree.lock() {
            tree.evict_tenant_by_size(max_size);
        }
    }
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker(
        &self,
        workers: &[Box<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Get current load statistics
        let loads: Vec<usize> = workers.iter().map(|w| w.load()).collect();
        let max_load = *loads.iter().max().unwrap_or(&0);
        let min_load = *loads.iter().min().unwrap_or(&0);

        // Check if load is imbalanced
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        if is_imbalanced {
            // Log load balancing trigger
            let worker_loads: Vec<(String, usize)> = workers
                .iter()
                .map(|w| (w.url().to_string(), w.load()))
                .collect();

            info!(
                "Load balancing triggered due to workload imbalance:\n\
                Max load: {}, Min load: {}\n\
                Current worker loads: {:?}",
                max_load, min_load, worker_loads
            );

            RouterMetrics::record_load_balancing_event();
            RouterMetrics::set_load_range(max_load, min_load);

            // Use shortest queue when imbalanced
            let min_load_idx = healthy_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())
                .copied()?;

            // Increment processed counter
            workers[min_load_idx].increment_processed();
            RouterMetrics::record_processed_request(workers[min_load_idx].url());

            return Some(min_load_idx);
        }

        // Use cache-aware routing when balanced
        let text = request_text.unwrap_or("");

        if let Ok(tree) = self.tree.lock() {
            let (matched_text, matched_worker) = tree.prefix_match(text);
            let match_rate = if text.is_empty() {
                0.0
            } else {
                matched_text.chars().count() as f32 / text.chars().count() as f32
            };

            let selected_url = if match_rate > self.config.cache_threshold {
                RouterMetrics::record_cache_hit();
                matched_worker.to_string()
            } else {
                RouterMetrics::record_cache_miss();
                tree.get_smallest_tenant()
            };

            // Find the index of the selected worker
            let selected_idx = workers.iter().position(|w| w.url() == selected_url)?;

            // Only proceed if the worker is healthy
            if !workers[selected_idx].is_healthy() {
                return healthy_indices.first().copied();
            }

            // Update the tree with this request
            tree.insert(text, &selected_url);

            // Increment processed counter
            workers[selected_idx].increment_processed();
            RouterMetrics::record_processed_request(&selected_url);

            return Some(selected_idx);
        }

        // Fallback to first healthy worker if tree operations fail
        healthy_indices.first().copied()
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Could track success rates per worker for more intelligent routing
        if !success {
            // Optionally reduce affinity for failed requests
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn select_worker_pair(
        &self,
        prefill_workers: &[Box<dyn Worker>],
        decode_workers: &[Box<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<(usize, usize)> {
        // In PD mode:
        // - Prefill: Use cache-aware routing for better cache utilization
        // - Decode: Use least-load routing for better load distribution

        // Select prefill worker using cache-aware logic
        let prefill_idx = self.select_worker(prefill_workers, request_text)?;

        // Select decode worker using least-load logic
        let healthy_decode = get_healthy_worker_indices(decode_workers);
        if healthy_decode.is_empty() {
            return None;
        }

        let decode_idx = healthy_decode
            .iter()
            .min_by_key(|&&idx| decode_workers[idx].load())
            .copied()?;

        Some((prefill_idx, decode_idx))
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CacheAwarePolicy {
    fn drop(&mut self) {
        // Note: We can't properly stop the eviction thread since it's in an infinite loop
        // In a production system, we'd use a channel or atomic flag to signal shutdown
        if let Some(handle) = self.eviction_handle.take() {
            // The thread will continue running until the program exits
            // This is acceptable for now since the router typically runs for the lifetime of the program
            drop(handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
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

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy.select_worker(&workers, Some("hello world")).unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy.select_worker(&workers, Some("hello world")).unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy.select_worker(&workers, Some("hello")).unwrap();
        assert_eq!(idx1, idx3);
    }

    #[test]
    fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
        });

        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Box<dyn Worker>> = vec![Box::new(worker1), Box::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, Some("test")).unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    #[test]
    fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
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

        policy.init_workers(&workers);

        // Route some requests
        policy.select_worker(&workers, Some("test1"));
        policy.select_worker(&workers, Some("test2"));

        // Remove a worker
        policy.remove_worker("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy.select_worker(&workers, Some("test1")).unwrap();
        assert_eq!(idx, 1);
    }
}
