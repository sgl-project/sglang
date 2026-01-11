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
//!
//! ## Redis Backend
//! When `SMG_MANUAL_REDIS_URL` environment variable is set, uses Redis for
//! distributed routing state. Otherwise uses local DashMap.

use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use dashmap::{mapref::entry::Entry, DashMap};
use rand::Rng;
use tracing::{error, info, warn};

use super::{
    get_healthy_worker_indices, utils::PeriodicTask, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::{
    config::ManualAssignmentMode, core::Worker, observability::metrics::Metrics,
    routers::header_utils::extract_routing_key,
};

const MAX_CANDIDATE_WORKERS: usize = 2;
const REDIS_KEY_PREFIX: &str = "smg:manual:";

// ------------------------------------ API layer ---------------------------------------

#[derive(Debug)]
pub struct ManualPolicy {
    backend: Backend,
    assignment_mode: ManualAssignmentMode,
}

impl Default for ManualPolicy {
    fn default() -> Self {
        Self::new()
    }
}

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

impl ManualPolicy {
    pub fn new() -> Self {
        Self::with_config(ManualConfig::default())
    }

    pub fn with_config(config: ManualConfig) -> Self {
        let backend = Backend::from_env(&config);
        Self {
            backend,
            assignment_mode: config.assignment_mode,
        }
    }

    async fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, ExecutionBranch) {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return (None, ExecutionBranch::NoHealthyWorkers);
        }

        if let Some(routing_id) = extract_routing_key(info.headers) {
            let (idx, branch) = self.backend.select_by_routing_id(
                routing_id,
                workers,
                &healthy_indices,
                self.assignment_mode,
            ).await;
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
    async fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo<'_>) -> Option<usize> {
        let (result, branch) = self.select_worker_impl(workers, info).await;
        Metrics::record_worker_manual_policy_branch(branch.as_str());
        if let Some(len) = self.backend.len() {
            Metrics::set_manual_policy_cache_entries(len);
        }
        result
    }

    fn name(&self) -> &'static str {
        "manual"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ------------------------------------ base backend ---------------------------------------

#[derive(Debug)]
enum Backend {
    Local(LocalBackend),
    Redis(RedisBackend),
}

impl Backend {
    fn from_env(config: &ManualConfig) -> Self {
        if let Ok(redis_url) = std::env::var("SMG_MANUAL_REDIS_URL") {
            let backend = RedisBackend::new(&redis_url, config.max_idle_secs)
                .expect("SMG_MANUAL_REDIS_URL is set but failed to connect to Redis");
            info!("ManualPolicy using Redis backend: {}", redis_url);
            return Backend::Redis(backend);
        }
        info!("ManualPolicy using local DashMap backend");
        Backend::Local(LocalBackend::new(config))
    }

    async fn select_by_routing_id(
        &self,
        routing_id: &str,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        assignment_mode: ManualAssignmentMode,
    ) -> (usize, ExecutionBranch) {
        match self {
            Backend::Local(b) => b.select_by_routing_id(routing_id, workers, healthy_indices, assignment_mode),
            Backend::Redis(b) => b.select_by_routing_id(routing_id, workers, healthy_indices, assignment_mode).await,
        }
    }

    fn len(&self) -> Option<usize> {
        match self {
            Backend::Local(b) => Some(b.len()),
            Backend::Redis(_) => None,
        }
    }
}

// ------------------------------------ local backend ---------------------------------------

#[derive(Debug)]
struct LocalBackend {
    routing_map: Arc<DashMap<String, LocalNode>>,
    _eviction_task: Option<PeriodicTask>,
}

#[derive(Debug, Clone)]
struct LocalNode {
    candi_worker_urls: Vec<String>,
    last_access: Instant,
}

impl LocalNode {
    fn push_bounded(&mut self, url: String) {
        while self.candi_worker_urls.len() >= MAX_CANDIDATE_WORKERS {
            self.candi_worker_urls.remove(0);
        }
        self.candi_worker_urls.push(url);
    }
}

impl LocalBackend {
    fn new(config: &ManualConfig) -> Self {
        use std::time::Duration;

        let routing_map = Arc::new(DashMap::<String, LocalNode>::new());

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
            _eviction_task: eviction_task,
        }
    }

    fn select_by_routing_id(
        &self,
        routing_id: &str,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        assignment_mode: ManualAssignmentMode,
    ) -> (usize, ExecutionBranch) {
        match self.routing_map.entry(routing_id.to_string()) {
            Entry::Occupied(mut entry) => {
                let node = entry.get_mut();
                node.last_access = Instant::now();
                if let Some(idx) = find_healthy_worker(&node.candi_worker_urls, workers, healthy_indices) {
                    (idx, ExecutionBranch::OccupiedHit)
                } else {
                    let selected_idx = select_new_worker(workers, healthy_indices, assignment_mode);
                    node.push_bounded(workers[selected_idx].url().to_string());
                    (selected_idx, ExecutionBranch::OccupiedMiss)
                }
            }
            Entry::Vacant(entry) => {
                let selected_idx = select_new_worker(workers, healthy_indices, assignment_mode);
                entry.insert(LocalNode {
                    candi_worker_urls: vec![workers[selected_idx].url().to_string()],
                    last_access: Instant::now(),
                });
                (selected_idx, ExecutionBranch::Vacant)
            }
        }
    }

    fn len(&self) -> usize {
        self.routing_map.len()
    }
}

// ------------------------------------ redis backend ---------------------------------------

#[derive(Debug)]
struct RedisBackend {
    pool: deadpool_redis::Pool,
    ttl_secs: u64,
}

impl RedisBackend {
    fn new(url: &str, ttl_secs: u64) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let cfg = deadpool_redis::Config::from_url(url);
        let pool = cfg.create_pool(Some(deadpool_redis::Runtime::Tokio1))?;
        Ok(Self { pool, ttl_secs })
    }

    fn key(&self, routing_id: &str) -> String {
        format!("{}{}", REDIS_KEY_PREFIX, routing_id)
    }

    fn parse_candidates(data: &str) -> Vec<String> {
        data.split(',').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
    }

    fn serialize_candidates(urls: &[String]) -> String {
        urls.join(",")
    }

    async fn get_conn_with_retry(&self) -> Option<deadpool_redis::Connection> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY_MS: u64 = 10;

        for attempt in 0..MAX_RETRIES {
            match self.pool.get().await {
                Ok(c) => return Some(c),
                Err(e) => {
                    if attempt < MAX_RETRIES - 1 {
                        warn!("Redis connection attempt {} failed: {}, retrying...", attempt + 1, e);
                        tokio::time::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS * (attempt as u64 + 1))).await;
                    } else {
                        error!("Redis connection failed after {} attempts: {}", MAX_RETRIES, e);
                    }
                }
            }
        }
        None
    }

    async fn select_by_routing_id(
        &self,
        routing_id: &str,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        assignment_mode: ManualAssignmentMode,
    ) -> (usize, ExecutionBranch) {
        let key = self.key(routing_id);

        let mut conn = match self.get_conn_with_retry().await {
            Some(c) => c,
            None => {
                return (random_select(healthy_indices), ExecutionBranch::Vacant);
            }
        };

        let existing: Option<String> = match redis::cmd("GET").arg(&key).query_async(&mut conn).await {
            Ok(v) => v,
            Err(e) => {
                warn!("Redis GET failed: {}", e);
                return (random_select(healthy_indices), ExecutionBranch::Vacant);
            }
        };

        if let Some(data) = existing {
            let candidates = Self::parse_candidates(&data);
            if let Some(idx) = find_healthy_worker(&candidates, workers, healthy_indices) {
                let _ = redis::cmd("EXPIRE").arg(&key).arg(self.ttl_secs).query_async::<()>(&mut conn).await;
                return (idx, ExecutionBranch::OccupiedHit);
            }
            let selected_idx = select_new_worker(workers, healthy_indices, assignment_mode);
            let new_url = workers[selected_idx].url();
            let mut new_candidates = candidates;
            while new_candidates.len() >= MAX_CANDIDATE_WORKERS {
                new_candidates.remove(0);
            }
            new_candidates.push(new_url.to_string());
            let new_data = Self::serialize_candidates(&new_candidates);
            let _ = redis::cmd("SET").arg(&key).arg(&new_data).arg("EX").arg(self.ttl_secs).query_async::<()>(&mut conn).await;
            return (selected_idx, ExecutionBranch::OccupiedMiss);
        }

        let selected_idx = select_new_worker(workers, healthy_indices, assignment_mode);
        let new_url = workers[selected_idx].url();

        let set_result: Result<Option<String>, _> = redis::cmd("SET")
            .arg(&key)
            .arg(new_url)
            .arg("NX")
            .arg("EX")
            .arg(self.ttl_secs)
            .arg("GET")
            .query_async(&mut conn)
            .await;

        match set_result {
            Ok(None) => (selected_idx, ExecutionBranch::Vacant),
            Ok(Some(existing_data)) => {
                let candidates = Self::parse_candidates(&existing_data);
                if let Some(idx) = find_healthy_worker(&candidates, workers, healthy_indices) {
                    (idx, ExecutionBranch::OccupiedHit)
                } else {
                    (selected_idx, ExecutionBranch::OccupiedMiss)
                }
            }
            Err(e) => {
                warn!("Redis SET NX failed: {}", e);
                (selected_idx, ExecutionBranch::Vacant)
            }
        }
    }
}

// ------------------------------------ util functions ---------------------------------------

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

fn select_new_worker(
    workers: &[Arc<dyn Worker>],
    healthy_indices: &[usize],
    assignment_mode: ManualAssignmentMode,
) -> usize {
    match assignment_mode {
        ManualAssignmentMode::Random => random_select(healthy_indices),
        ManualAssignmentMode::MinLoad => min_load_select(workers, healthy_indices),
        ManualAssignmentMode::MinGroup => min_group_select(workers, healthy_indices),
    }
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

    #[tokio::test]
    async fn test_manual_consistent_routing() {
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

    #[tokio::test]
    async fn test_manual_different_routing_ids() {
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

    #[tokio::test]
    async fn test_manual_fallback_random() {
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

    #[tokio::test]
    async fn test_manual_with_unhealthy_workers() {
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

    #[tokio::test]
    async fn test_manual_no_healthy_workers() {
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

    #[tokio::test]
    async fn test_manual_empty_routing_id() {
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

    #[tokio::test]
    async fn test_manual_remaps_when_worker_becomes_unhealthy() {
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

    #[tokio::test]
    async fn test_manual_empty_workers() {
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

    #[tokio::test]
    async fn test_manual_single_worker() {
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

    #[tokio::test]
    async fn test_manual_worker_recovery() {
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

    #[tokio::test]
    async fn test_manual_max_candidate_workers_eviction() {
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

    #[tokio::test]
    async fn test_manual_policy_name() {
        let policy = ManualPolicy::new();
        assert_eq!(policy.name(), "manual");
    }

    #[tokio::test]
    async fn test_manual_routing_info_push_bounded() {
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

    #[tokio::test]
    async fn test_manual_find_healthy_worker_priority() {
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

    #[tokio::test]
    async fn test_manual_find_worker_index_by_url() {
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

    #[tokio::test]
    async fn test_manual_config_default() {
        let config = ManualConfig::default();
        assert_eq!(config.eviction_interval_secs, 60);
        assert_eq!(config.max_idle_secs, 4 * 3600);
    }

    #[tokio::test]
    async fn test_manual_with_disabled_eviction() {
        let config = ManualConfig {
            eviction_interval_secs: 0,
            max_idle_secs: 3600,
            assignment_mode: ManualAssignmentMode::Random,
        };
        let policy = ManualPolicy::with_config(config);
        assert!(policy._eviction_task.is_none());
    }

    #[tokio::test]
    async fn test_manual_last_access_updates() {
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

    #[tokio::test]
    async fn test_manual_ttl_eviction_logic() {
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

    #[tokio::test]
    async fn test_min_group_select_distributes_evenly() {
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

    #[tokio::test]
    async fn test_min_group_select_prefers_worker_with_fewer_routing_keys() {
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

    #[tokio::test]
    async fn test_min_load_select_prefers_worker_with_fewer_requests() {
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

    #[tokio::test]
    async fn test_min_group_sticky_after_assignment() {
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

    #[tokio::test]
    async fn test_random_mode_does_not_consider_load() {
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
}
