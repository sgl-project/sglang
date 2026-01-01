//! Manual routing policy based on routing key header
//!
//! This policy provides sticky session routing where each unique routing key
//! is consistently mapped to the same worker. Unlike consistent hashing,
//! this policy:
//! - Does NOT redistribute any sessions when workers are added
//! - Only remaps sessions when their assigned worker becomes unhealthy
//! - Maintains up to 2 candidate workers per routing key for fast failover
//!
//! Use this when you need stronger stickiness guarantees than consistent hashing,
//! for example with stateful chat sessions where context is stored on the worker.
//!
//! ## Header
//! - `X-SMG-Routing-Key`: The routing key for sticky session routing

use std::{sync::Arc, time::Instant};

use dashmap::{mapref::entry::Entry, DashMap};
use http::header::HeaderName;
use rand::Rng;
use tracing::debug;

use super::{get_healthy_worker_indices, utils::PeriodicTask, LoadBalancingPolicy, SelectWorkerInfo};
use crate::{core::Worker, observability::metrics::Metrics};

/// Header for routing key based sticky sessions
static HEADER_ROUTING_KEY: HeaderName = HeaderName::from_static("x-smg-routing-key");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionBranch {
    NoHealthyWorkers,
    FastPathHit,
    SlowPathOccupiedHit,
    SlowPathOccupiedMiss,
    SlowPathVacant,
    NoRoutingId,
}

impl ExecutionBranch {
    fn as_str(&self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::FastPathHit => "fast_path_hit",
            Self::SlowPathOccupiedHit => "slow_path_occupied_hit",
            Self::SlowPathOccupiedMiss => "slow_path_occupied_miss",
            Self::SlowPathVacant => "slow_path_vacant",
            Self::NoRoutingId => "no_routing_id",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RoutingId(String);

impl RoutingId {
    fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

const MAX_CANDIDATE_WORKERS: usize = 2;

#[derive(Debug, Clone)]
pub struct ManualConfig {
    pub eviction_interval_secs: u64,
    pub max_entries: usize,
}

impl Default for ManualConfig {
    fn default() -> Self {
        Self {
            eviction_interval_secs: 60,
            max_entries: 10000,
        }
    }
}

#[derive(Debug, Clone)]
struct Node {
    candi_worker_urls: Vec<String>,
    last_access: Instant,
}

impl Node {
    fn push_bounded(&mut self, url: String) {
        while self.candi_worker_urls.len() >= MAX_CANDIDATE_WORKERS {
            self.candi_worker_urls.remove(0);
        }
        self.candi_worker_urls.push(url);
    }
}

// TODO may optimize performance
#[derive(Debug)]
pub struct ManualPolicy {
    #[allow(dead_code)]
    config: ManualConfig,
    routing_map: Arc<DashMap<RoutingId, Node>>,
    _eviction_task: Option<PeriodicTask>,
}

impl Default for ManualPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl ManualPolicy {
    pub fn new() -> Self {
        Self::with_config(ManualConfig::default())
    }

    pub fn with_config(config: ManualConfig) -> Self {
        let routing_map = Arc::new(DashMap::<RoutingId, Node>::new());

        let eviction_task = if config.eviction_interval_secs > 0 && config.max_entries > 0 {
            let routing_map_clone = Arc::clone(&routing_map);
            let max_entries = config.max_entries;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "ManualPolicyEviction",
                move || {
                    let current_size = routing_map_clone.len();
                    if current_size <= max_entries {
                        return;
                    }

                    let to_evict = current_size - max_entries;

                    // Collect entries with their last_access time
                    let mut entries: Vec<_> = routing_map_clone
                        .iter()
                        .map(|entry| (entry.key().clone(), entry.value().last_access))
                        .collect();

                    // Sort by last_access (oldest first)
                    entries.sort_by_key(|(_, last_access)| *last_access);

                    // Remove oldest entries
                    let evicted_count = entries
                        .iter()
                        .take(to_evict)
                        .filter(|(key, _)| routing_map_clone.remove(key).is_some())
                        .count();

                    debug!(
                        "ManualPolicy LRU eviction: evicted {} entries, remaining {} (max: {})",
                        evicted_count,
                        routing_map_clone.len(),
                        max_entries
                    );
                },
            ))
        } else {
            None
        };

        Self {
            config,
            routing_map,
            _eviction_task: eviction_task,
        }
    }

    fn select_by_routing_id(
        &self,
        workers: &[Arc<dyn Worker>],
        routing_id: &str,
        healthy_indices: &[usize],
    ) -> (usize, ExecutionBranch) {
        let routing_id = RoutingId::new(routing_id);

        // Fast path: read-only check first (shared lock)
        if let Some(info) = self.routing_map.get(&routing_id) {
            if let Some(idx) =
                find_healthy_worker(&info.candi_worker_urls, workers, healthy_indices)
            {
                drop(info); // Release read lock before acquiring write lock
                // Update last_access with write lock
                if let Some(mut info) = self.routing_map.get_mut(&routing_id) {
                    info.last_access = Instant::now();
                }
                return (idx, ExecutionBranch::FastPathHit);
            }
        }

        // Slow path
        match self.routing_map.entry(routing_id) {
            Entry::Occupied(mut entry) => {
                if let Some(idx) =
                    find_healthy_worker(&entry.get().candi_worker_urls, workers, healthy_indices)
                {
                    entry.get_mut().last_access = Instant::now();
                    return (idx, ExecutionBranch::SlowPathOccupiedHit);
                }
                let selected_idx = random_select(healthy_indices);
                let node = entry.get_mut();
                node.push_bounded(workers[selected_idx].url().to_string());
                node.last_access = Instant::now();
                (selected_idx, ExecutionBranch::SlowPathOccupiedMiss)
            }
            Entry::Vacant(entry) => {
                let selected_idx = random_select(healthy_indices);
                entry.insert(Node {
                    candi_worker_urls: vec![workers[selected_idx].url().to_string()],
                    last_access: Instant::now(),
                });
                (selected_idx, ExecutionBranch::SlowPathVacant)
            }
        }
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, ExecutionBranch) {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return (None, ExecutionBranch::NoHealthyWorkers);
        }

        // Extract routing key from header
        let routing_id = info
            .headers
            .and_then(|h| h.get(&HEADER_ROUTING_KEY))
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty());

        if let Some(routing_id) = routing_id {
            let (idx, branch) = self.select_by_routing_id(workers, routing_id, &healthy_indices);
            return (Some(idx), branch);
        }

        (
            Some(random_select(&healthy_indices)),
            ExecutionBranch::NoRoutingId,
        )
    }
}

impl LoadBalancingPolicy for ManualPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let (result, branch) = self.select_worker_impl(workers, info);
        Metrics::record_worker_manual_policy_branch(branch.as_str());
        Metrics::set_manual_policy_cache_entries(self.routing_map.len());
        result
    }

    fn name(&self) -> &'static str {
        "manual"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn find_healthy_worker(
    urls: &[String],
    workers: &[Arc<dyn Worker>],
    healthy_indices: &[usize],
) -> Option<usize> {
    for url in urls {
        if let Some(idx) = find_worker_index_by_url(workers, url) {
            if healthy_indices.contains(&idx) {
                return Some(idx);
            }
        }
    }
    None
}

fn find_worker_index_by_url(workers: &[Arc<dyn Worker>], url: &str) -> Option<usize> {
    workers.iter().position(|w| w.url() == url)
}

// TODO: use load-aware selection later
fn random_select(healthy_indices: &[usize]) -> usize {
    let mut rng = rand::rng();
    let random_idx = rng.random_range(0..healthy_indices.len());
    healthy_indices[random_idx]
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn create_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    fn headers_with_routing_key(key: &str) -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", key.parse().unwrap());
        headers
    }

    #[test]
    fn test_manual_consistent_routing() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let headers = headers_with_routing_key("user-123");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(
                result,
                Some(first_idx),
                "Same routing_id should route to same worker"
            );
            assert_eq!(branch, ExecutionBranch::FastPathHit);
        }
    }

    #[test]
    fn test_manual_different_routing_ids() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let mut distribution = HashMap::new();
        for i in 0..100 {
            let headers = headers_with_routing_key(&format!("user-{}", i));
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(branch, ExecutionBranch::SlowPathVacant);
            *distribution.entry(result.unwrap()).or_insert(0) += 1;
        }

        assert!(
            distribution.len() > 1,
            "Should distribute across multiple workers"
        );
    }

    #[test]
    fn test_manual_fallback_random() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let mut counts = HashMap::new();
        for _ in 0..100 {
            let info = SelectWorkerInfo::default();
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(branch, ExecutionBranch::NoRoutingId);
            if let Some(idx) = result {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        assert_eq!(counts.len(), 2, "Random fallback should use all workers");
    }

    #[test]
    fn test_manual_with_unhealthy_workers() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        workers[0].set_healthy(false);

        let headers = headers_with_routing_key("test-routing-id");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(1), "Should only select healthy worker");
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(1), "Should only select healthy worker");
            assert_eq!(branch, ExecutionBranch::FastPathHit);
        }
    }

    #[test]
    fn test_manual_no_healthy_workers() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000"]);

        workers[0].set_healthy(false);
        let headers = headers_with_routing_key("test");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, ExecutionBranch::NoHealthyWorkers);
    }

    #[test]
    fn test_manual_empty_routing_id() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let mut counts = HashMap::new();
        for _ in 0..100 {
            let headers = headers_with_routing_key("");
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(branch, ExecutionBranch::NoRoutingId);
            if let Some(idx) = result {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        assert_eq!(
            counts.len(),
            2,
            "Empty routing_id should use random fallback"
        );
    }

    #[test]
    fn test_manual_remaps_when_worker_becomes_unhealthy() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_routing_key("sticky-user");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        workers[first_idx].set_healthy(false);

        let (new_result, branch) = policy.select_worker_impl(&workers, &info);
        let new_idx = new_result.unwrap();
        assert_ne!(new_idx, first_idx, "Should remap to healthy worker");
        assert_eq!(branch, ExecutionBranch::SlowPathOccupiedMiss);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(
                result,
                Some(new_idx),
                "Should consistently route to new worker"
            );
            assert_eq!(branch, ExecutionBranch::FastPathHit);
        }
    }

    #[test]
    fn test_manual_empty_workers() {
        let policy = ManualPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];
        let headers = headers_with_routing_key("test");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, ExecutionBranch::NoHealthyWorkers);
    }

    #[test]
    fn test_manual_single_worker() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000"]);

        let headers = headers_with_routing_key("single-test");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, Some(0));
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(0));
            assert_eq!(branch, ExecutionBranch::FastPathHit);
        }
    }

    #[test]
    fn test_manual_worker_recovery() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_routing_key("recovery-test");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        workers[first_idx].set_healthy(false);

        let (second_result, branch) = policy.select_worker_impl(&workers, &info);
        let second_idx = second_result.unwrap();
        assert_ne!(second_idx, first_idx);
        assert_eq!(branch, ExecutionBranch::SlowPathOccupiedMiss);

        workers[first_idx].set_healthy(true);

        let (after_recovery, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(
            after_recovery,
            Some(first_idx),
            "Should return to original worker after recovery since it's first in candidate list"
        );
        assert_eq!(branch, ExecutionBranch::FastPathHit);
    }

    #[test]
    fn test_manual_max_candidate_workers_eviction() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let headers = headers_with_routing_key("eviction-test");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::SlowPathVacant);

        workers[first_idx].set_healthy(false);

        let (second_result, branch) = policy.select_worker_impl(&workers, &info);
        let second_idx = second_result.unwrap();
        assert_ne!(second_idx, first_idx);
        assert_eq!(branch, ExecutionBranch::SlowPathOccupiedMiss);

        workers[second_idx].set_healthy(false);

        let remaining_idx = (0..3).find(|&i| i != first_idx && i != second_idx).unwrap();
        let (third_result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(
            third_result,
            Some(remaining_idx),
            "Should select the only remaining healthy worker"
        );
        assert_eq!(branch, ExecutionBranch::SlowPathOccupiedMiss);

        workers[first_idx].set_healthy(true);

        let (idx_after_restore, branch) = policy.select_worker_impl(&workers, &info);
        assert_ne!(
            idx_after_restore,
            Some(first_idx),
            "First worker should be evicted from candidates due to MAX_CANDIDATE_WORKERS=2"
        );
        assert_eq!(branch, ExecutionBranch::FastPathHit);
    }

    #[test]
    fn test_manual_policy_name() {
        let policy = ManualPolicy::new();
        assert_eq!(policy.name(), "manual");
    }

    #[test]
    fn test_manual_routing_info_push_bounded() {
        let mut info = Node {
            candi_worker_urls: vec!["http://w1:8000".to_string()],
            last_access: Instant::now(),
        };

        info.push_bounded("http://w2:8000".to_string());
        assert_eq!(info.candi_worker_urls.len(), 2);
        assert_eq!(info.candi_worker_urls[0], "http://w1:8000");
        assert_eq!(info.candi_worker_urls[1], "http://w2:8000");

        info.push_bounded("http://w3:8000".to_string());
        assert_eq!(info.candi_worker_urls.len(), 2);
        assert_eq!(
            info.candi_worker_urls[0], "http://w2:8000",
            "Oldest entry should be removed"
        );
        assert_eq!(info.candi_worker_urls[1], "http://w3:8000");
    }

    #[test]
    fn test_manual_find_healthy_worker_priority() {
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let urls = vec![
            "http://w1:8000".to_string(),
            "http://w2:8000".to_string(),
            "http://w3:8000".to_string(),
        ];
        let healthy_indices = vec![0, 1, 2];

        let result = find_healthy_worker(&urls, &workers, &healthy_indices);
        assert_eq!(
            result,
            Some(0),
            "Should return first healthy worker in urls"
        );

        workers[0].set_healthy(false);
        let healthy_indices = vec![1, 2];
        let result = find_healthy_worker(&urls, &workers, &healthy_indices);
        assert_eq!(result, Some(1), "Should skip unhealthy and return next");

        workers[1].set_healthy(false);
        let healthy_indices = vec![2];
        let result = find_healthy_worker(&urls, &workers, &healthy_indices);
        assert_eq!(result, Some(2), "Should return last healthy worker");

        workers[2].set_healthy(false);
        let healthy_indices: Vec<usize> = vec![];
        let result = find_healthy_worker(&urls, &workers, &healthy_indices);
        assert_eq!(result, None, "Should return None when no healthy workers");
    }

    #[test]
    fn test_manual_find_worker_index_by_url() {
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        assert_eq!(
            find_worker_index_by_url(&workers, "http://w1:8000"),
            Some(0)
        );
        assert_eq!(
            find_worker_index_by_url(&workers, "http://w2:8000"),
            Some(1)
        );
        assert_eq!(
            find_worker_index_by_url(&workers, "http://w3:8000"),
            None,
            "Should return None for unknown URL"
        );
    }

    #[test]
    fn test_manual_config_default() {
        let config = ManualConfig::default();
        assert_eq!(config.eviction_interval_secs, 60);
        assert_eq!(config.max_entries, 10000);
    }

    #[test]
    fn test_manual_with_config() {
        let config = ManualConfig {
            eviction_interval_secs: 30,
            max_entries: 5000,
        };
        let policy = ManualPolicy::with_config(config);
        assert_eq!(policy.config.eviction_interval_secs, 30);
        assert_eq!(policy.config.max_entries, 5000);
    }

    #[test]
    fn test_manual_with_disabled_eviction() {
        let config = ManualConfig {
            eviction_interval_secs: 0,
            max_entries: 10000,
        };
        let policy = ManualPolicy::with_config(config);
        assert!(policy._eviction_task.is_none());
    }

    #[test]
    fn test_manual_last_access_updates_on_access() {
        let config = ManualConfig {
            eviction_interval_secs: 0,
            max_entries: 10000,
        };
        let policy = ManualPolicy::with_config(config);
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_routing_key("test-key");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        policy.select_worker_impl(&workers, &info);

        let routing_id = RoutingId::new("test-key");
        let node = policy.routing_map.get(&routing_id).unwrap();
        let first_access = node.last_access;

        std::thread::sleep(std::time::Duration::from_millis(10));

        policy.select_worker_impl(&workers, &info);

        let node = policy.routing_map.get(&routing_id).unwrap();
        assert!(
            node.last_access > first_access,
            "last_access should be updated on subsequent access"
        );
    }

    #[test]
    fn test_manual_lru_eviction_by_size() {
        use std::thread;
        use std::time::Duration;

        let config = ManualConfig {
            eviction_interval_secs: 1,
            max_entries: 2,
        };
        let policy = ManualPolicy::with_config(config);
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        // Add 3 entries with small delays to ensure different last_access times
        for i in 0..3 {
            let headers = headers_with_routing_key(&format!("key-{}", i));
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            policy.select_worker_impl(&workers, &info);
            thread::sleep(Duration::from_millis(50));
        }

        assert_eq!(policy.routing_map.len(), 3, "Should have 3 entries before eviction");

        // Wait for eviction to run
        thread::sleep(Duration::from_secs(2));

        assert_eq!(
            policy.routing_map.len(),
            2,
            "Should have 2 entries after LRU eviction (max_entries=2)"
        );

        // The oldest entry (key-0) should be evicted
        let routing_id = RoutingId::new("key-0");
        assert!(
            policy.routing_map.get(&routing_id).is_none(),
            "Oldest entry should be evicted"
        );
    }
}
