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

use async_trait::async_trait;
use dashmap::{mapref::entry::Entry, DashMap};
use rand::Rng;
use tracing::info;

use super::{
    get_healthy_worker_indices, utils::PeriodicTask, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::{
    config::ManualAssignmentMode, core::Worker, observability::metrics::Metrics,
    routers::header_utils::extract_routing_key,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionBranch {
    NoHealthyWorkers,
    OccupiedHit,
    OccupiedMiss,
    Vacant,
    NoRoutingId,
}

impl ExecutionBranch {
    fn as_str(&self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::OccupiedHit => "occupied_hit",
            Self::OccupiedMiss => "occupied_miss",
            Self::Vacant => "vacant",
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
    pub max_idle_secs: u64,
    pub assignment_mode: ManualAssignmentMode,
}

impl Default for ManualConfig {
    fn default() -> Self {
        Self {
            eviction_interval_secs: 60,
            max_idle_secs: 4 * 3600,
            assignment_mode: ManualAssignmentMode::Random,
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

#[derive(Debug)]
pub struct ManualPolicy {
    routing_map: Arc<DashMap<RoutingId, Node>>,
    assignment_mode: ManualAssignmentMode,
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
        use std::time::Duration;

        let routing_map = Arc::new(DashMap::<RoutingId, Node>::new());

        let eviction_task = if config.eviction_interval_secs > 0 && config.max_idle_secs > 0 {
            let routing_map_clone = Arc::clone(&routing_map);
            let max_idle = Duration::from_secs(config.max_idle_secs);

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "ManualPolicyEviction",
                move || {
                    let now = Instant::now();
                    let before_size = routing_map_clone.len();

                    routing_map_clone
                        .retain(|_, node| now.duration_since(node.last_access) < max_idle);

                    let evicted_count = before_size - routing_map_clone.len();
                    if evicted_count > 0 {
                        info!(
                            "ManualPolicy TTL eviction: evicted {} entries, remaining {} (max_idle: {}s)",
                            evicted_count,
                            routing_map_clone.len(),
                            max_idle.as_secs()
                        );
                    }
                },
            ))
        } else {
            None
        };

        Self {
            routing_map,
            assignment_mode: config.assignment_mode,
            _eviction_task: eviction_task,
        }
    }

    fn select_new_worker(&self, workers: &[Arc<dyn Worker>], healthy_indices: &[usize]) -> usize {
        match self.assignment_mode {
            ManualAssignmentMode::Random => random_select(healthy_indices),
            ManualAssignmentMode::MinLoad => min_load_select(workers, healthy_indices),
            ManualAssignmentMode::MinGroup => min_group_select(workers, healthy_indices),
        }
    }

    fn select_by_routing_id(
        &self,
        workers: &[Arc<dyn Worker>],
        routing_id: &str,
        healthy_indices: &[usize],
    ) -> (usize, ExecutionBranch) {
        let routing_id = RoutingId::new(routing_id);

        match self.routing_map.entry(routing_id) {
            Entry::Occupied(mut entry) => {
                let node = entry.get_mut();
                node.last_access = Instant::now();
                if let Some(idx) =
                    find_healthy_worker(&node.candi_worker_urls, workers, healthy_indices)
                {
                    (idx, ExecutionBranch::OccupiedHit)
                } else {
                    let selected_idx = self.select_new_worker(workers, healthy_indices);
                    node.push_bounded(workers[selected_idx].url().to_string());
                    (selected_idx, ExecutionBranch::OccupiedMiss)
                }
            }
            Entry::Vacant(entry) => {
                let selected_idx = self.select_new_worker(workers, healthy_indices);
                entry.insert(Node {
                    candi_worker_urls: vec![workers[selected_idx].url().to_string()],
                    last_access: Instant::now(),
                });
                (selected_idx, ExecutionBranch::Vacant)
            }
        }
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, ExecutionBranch) {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return (None, ExecutionBranch::NoHealthyWorkers);
        }

        if let Some(routing_id) = extract_routing_key(info.headers) {
            let (idx, branch) = self.select_by_routing_id(workers, routing_id, &healthy_indices);
            return (Some(idx), branch);
        }

        (
            Some(random_select(&healthy_indices)),
            ExecutionBranch::NoRoutingId,
        )
    }
}

#[async_trait]
impl LoadBalancingPolicy for ManualPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
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

fn random_select(healthy_indices: &[usize]) -> usize {
    let mut rng = rand::rng();
    let random_idx = rng.random_range(0..healthy_indices.len());
    healthy_indices[random_idx]
}

fn select_min_by<K, V, F>(indices: &[K], get_value: F) -> K
where
    K: Copy,
    V: Ord,
    F: Fn(K) -> V,
{
    let mut min_val: Option<V> = None;
    let mut candidates = Vec::new();

    for &idx in indices {
        let val = get_value(idx);
        match min_val.as_ref().map(|m| val.cmp(m)) {
            None | Some(std::cmp::Ordering::Less) => {
                min_val = Some(val);
                candidates.clear();
                candidates.push(idx);
            }
            Some(std::cmp::Ordering::Equal) => {
                candidates.push(idx);
            }
            Some(std::cmp::Ordering::Greater) => {}
        }
    }

    if candidates.len() == 1 {
        candidates[0]
    } else {
        let mut rng = rand::rng();
        candidates[rng.random_range(0..candidates.len())]
    }
}

fn min_load_select(workers: &[Arc<dyn Worker>], healthy_indices: &[usize]) -> usize {
    select_min_by(healthy_indices, |idx| workers[idx].load())
}

fn min_group_select(workers: &[Arc<dyn Worker>], healthy_indices: &[usize]) -> usize {
    select_min_by(healthy_indices, |idx| {
        workers[idx].worker_routing_key_load().value()
    })
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
        assert_eq!(branch, ExecutionBranch::Vacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(
                result,
                Some(first_idx),
                "Same routing_id should route to same worker"
            );
            assert_eq!(branch, ExecutionBranch::OccupiedHit);
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
            assert_eq!(branch, ExecutionBranch::Vacant);
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
        assert_eq!(branch, ExecutionBranch::Vacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(1), "Should only select healthy worker");
            assert_eq!(branch, ExecutionBranch::OccupiedHit);
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
        assert_eq!(branch, ExecutionBranch::Vacant);

        workers[first_idx].set_healthy(false);

        let (new_result, branch) = policy.select_worker_impl(&workers, &info);
        let new_idx = new_result.unwrap();
        assert_ne!(new_idx, first_idx, "Should remap to healthy worker");
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(
                result,
                Some(new_idx),
                "Should consistently route to new worker"
            );
            assert_eq!(branch, ExecutionBranch::OccupiedHit);
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
        assert_eq!(branch, ExecutionBranch::Vacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(0));
            assert_eq!(branch, ExecutionBranch::OccupiedHit);
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
        assert_eq!(branch, ExecutionBranch::Vacant);

        workers[first_idx].set_healthy(false);

        let (second_result, branch) = policy.select_worker_impl(&workers, &info);
        let second_idx = second_result.unwrap();
        assert_ne!(second_idx, first_idx);
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        workers[first_idx].set_healthy(true);

        let (after_recovery, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(
            after_recovery,
            Some(first_idx),
            "Should return to original worker after recovery since it's first in candidate list"
        );
        assert_eq!(branch, ExecutionBranch::OccupiedHit);
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
        assert_eq!(branch, ExecutionBranch::Vacant);

        workers[first_idx].set_healthy(false);

        let (second_result, branch) = policy.select_worker_impl(&workers, &info);
        let second_idx = second_result.unwrap();
        assert_ne!(second_idx, first_idx);
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        workers[second_idx].set_healthy(false);

        let remaining_idx = (0..3).find(|&i| i != first_idx && i != second_idx).unwrap();
        let (third_result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(
            third_result,
            Some(remaining_idx),
            "Should select the only remaining healthy worker"
        );
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        workers[first_idx].set_healthy(true);

        let (idx_after_restore, branch) = policy.select_worker_impl(&workers, &info);
        assert_ne!(
            idx_after_restore,
            Some(first_idx),
            "First worker should be evicted from candidates due to MAX_CANDIDATE_WORKERS=2"
        );
        assert_eq!(branch, ExecutionBranch::OccupiedHit);
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
        assert_eq!(config.max_idle_secs, 4 * 3600);
    }

    #[test]
    fn test_manual_with_disabled_eviction() {
        let config = ManualConfig {
            eviction_interval_secs: 0,
            max_idle_secs: 3600,
            assignment_mode: ManualAssignmentMode::Random,
        };
        let policy = ManualPolicy::with_config(config);
        assert!(policy._eviction_task.is_none());
    }

    #[test]
    fn test_manual_last_access_updates() {
        let policy = ManualPolicy::new();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
        let headers = headers_with_routing_key("test-key");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let routing_id = RoutingId::new("test-key");

        // Vacant: first access
        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(branch, ExecutionBranch::Vacant);
        let first_idx = result.unwrap();
        let access_after_vacant = policy.routing_map.get(&routing_id).unwrap().last_access;
        assert!(access_after_vacant.elapsed().as_millis() < 100);

        std::thread::sleep(std::time::Duration::from_millis(10));

        // OccupiedHit: same worker still healthy
        let (_, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(branch, ExecutionBranch::OccupiedHit);
        let access_after_hit = policy.routing_map.get(&routing_id).unwrap().last_access;
        assert!(access_after_hit > access_after_vacant);

        std::thread::sleep(std::time::Duration::from_millis(10));

        // OccupiedMiss: worker becomes unhealthy
        workers[first_idx].set_healthy(false);
        let (_, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);
        let access_after_miss = policy.routing_map.get(&routing_id).unwrap().last_access;
        assert!(access_after_miss > access_after_hit);
    }

    #[test]
    fn test_manual_ttl_eviction_logic() {
        use std::time::Duration;

        let config = ManualConfig {
            eviction_interval_secs: 2,
            max_idle_secs: 2,
            assignment_mode: ManualAssignmentMode::Random,
        };
        let policy = ManualPolicy::with_config(config);
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_routing_key("key-0");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        policy.select_worker_impl(&workers, &info);

        assert_eq!(policy.routing_map.len(), 1);

        std::thread::sleep(Duration::from_secs(4));

        assert_eq!(policy.routing_map.len(), 0);
    }

    #[test]
    fn test_min_group_select_distributes_evenly() {
        let config = ManualConfig {
            assignment_mode: ManualAssignmentMode::MinGroup,
            ..Default::default()
        };
        let policy = ManualPolicy::with_config(config);
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        for i in 0..9 {
            let routing_key = format!("key-{}", i);
            let headers = headers_with_routing_key(&routing_key);
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };

            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert!(result.is_some());
            assert_eq!(branch, ExecutionBranch::Vacant);

            let selected_idx = result.unwrap();
            workers[selected_idx]
                .worker_routing_key_load()
                .increment(&routing_key);
        }

        let distribution: HashMap<_, usize> = policy
            .routing_map
            .iter()
            .map(|e| e.candi_worker_urls.first().unwrap().clone())
            .fold(HashMap::new(), |mut acc, url| {
                *acc.entry(url).or_default() += 1;
                acc
            });

        assert_eq!(distribution.len(), 3, "Should use all 3 workers");
        for count in distribution.values() {
            assert_eq!(*count, 3, "Each worker should have exactly 3 routing keys");
        }
    }

    #[test]
    fn test_min_group_select_prefers_worker_with_fewer_routing_keys() {
        let config = ManualConfig {
            assignment_mode: ManualAssignmentMode::MinGroup,
            ..Default::default()
        };
        let policy = ManualPolicy::with_config(config);
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        workers[0].worker_routing_key_load().increment("existing-1");
        workers[0].worker_routing_key_load().increment("existing-2");
        workers[1].worker_routing_key_load().increment("existing-3");

        assert_eq!(workers[0].worker_routing_key_load().value(), 2);
        assert_eq!(workers[1].worker_routing_key_load().value(), 1);
        assert_eq!(workers[2].worker_routing_key_load().value(), 0);

        let headers = headers_with_routing_key("new-key");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let (result, _) = policy.select_worker_impl(&workers, &info);
        let selected_idx = result.unwrap();

        assert_eq!(selected_idx, 2, "Should select worker with 0 routing keys");
    }

    #[test]
    fn test_min_load_select_prefers_worker_with_fewer_requests() {
        let config = ManualConfig {
            assignment_mode: ManualAssignmentMode::MinLoad,
            ..Default::default()
        };
        let policy = ManualPolicy::with_config(config);
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        workers[0].increment_load();
        workers[0].increment_load();
        workers[1].increment_load();

        assert_eq!(workers[0].load(), 2);
        assert_eq!(workers[1].load(), 1);
        assert_eq!(workers[2].load(), 0);

        let headers = headers_with_routing_key("new-key");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let (result, _) = policy.select_worker_impl(&workers, &info);
        let selected_idx = result.unwrap();

        assert_eq!(selected_idx, 2, "Should select worker with 0 load");
    }

    #[test]
    fn test_min_group_sticky_after_assignment() {
        let config = ManualConfig {
            assignment_mode: ManualAssignmentMode::MinGroup,
            ..Default::default()
        };
        let policy = ManualPolicy::with_config(config);
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        workers[0].worker_routing_key_load().increment("key-0");
        workers[1].worker_routing_key_load().increment("key-1");
        workers[1].worker_routing_key_load().increment("key-2");

        let headers = headers_with_routing_key("new-key");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let (first_result, branch) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::Vacant);
        assert_eq!(
            first_idx, 0,
            "Should select worker 0 (has 1 routing key vs 2)"
        );

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(
                result,
                Some(first_idx),
                "Same routing key should route to same worker"
            );
            assert_eq!(branch, ExecutionBranch::OccupiedHit);
        }
    }

    #[test]
    fn test_random_mode_does_not_consider_load() {
        let config = ManualConfig {
            assignment_mode: ManualAssignmentMode::Random,
            ..Default::default()
        };
        let policy = ManualPolicy::with_config(config);
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        workers[0].worker_routing_key_load().increment("key-1");
        workers[0].worker_routing_key_load().increment("key-2");
        workers[0].worker_routing_key_load().increment("key-3");

        let mut selected_worker_0 = false;
        for i in 0..50 {
            let headers = headers_with_routing_key(&format!("test-{}", i));
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            let (result, _) = policy.select_worker_impl(&workers, &info);
            if result == Some(0) {
                selected_worker_0 = true;
                break;
            }
        }
        assert!(
            selected_worker_0,
            "Random mode should sometimes select worker 0 despite higher load"
        );
    }

    fn assert_no_routing_key_uses_random(
        assignment_mode: ManualAssignmentMode,
        setup_load: impl Fn(&[Arc<dyn Worker>]),
    ) {
        let config = ManualConfig {
            assignment_mode,
            ..Default::default()
        };
        let policy = ManualPolicy::with_config(config);
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);
        setup_load(&workers);

        let mut selected_worker_0 = false;
        for _ in 0..50 {
            let info = SelectWorkerInfo::default();
            let (result, branch) = policy.select_worker_impl(&workers, &info);
            assert_eq!(branch, ExecutionBranch::NoRoutingId);
            if result == Some(0) {
                selected_worker_0 = true;
                break;
            }
        }
        assert!(
            selected_worker_0,
            "Should randomly select worker 0 despite higher load"
        );
    }

    #[test]
    fn test_no_routing_key_uses_random_even_with_min_load_mode() {
        assert_no_routing_key_uses_random(ManualAssignmentMode::MinLoad, |workers| {
            workers[0].increment_load();
            workers[0].increment_load();
        });
    }

    #[test]
    fn test_no_routing_key_uses_random_even_with_min_group_mode() {
        assert_no_routing_key_uses_random(ManualAssignmentMode::MinGroup, |workers| {
            workers[0].worker_routing_key_load().increment("k1");
            workers[0].worker_routing_key_load().increment("k2");
        });
    }
}
