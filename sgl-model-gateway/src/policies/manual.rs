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
use redis::{AsyncCommands, Expiry};
use tracing::{info, warn};

use super::{
    get_healthy_worker_indices, utils::PeriodicTask, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::{
    config::{ManualAssignmentMode, RetryConfig},
    core::{
        retry::{MaxRetriesExceeded, RetryExecutor},
        Worker,
    },
    observability::metrics::Metrics,
    routers::header_utils::extract_routing_key,
};

const MAX_CANDIDATE_WORKERS: usize = 2;
const REDIS_KEY_PREFIX: &str = "smg:manual_policy:";

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
    NoRoutingId,
    OccupiedHit,
    OccupiedMiss,
    Vacant,
    RedisPoolGetException,
    RedisGetexException,
    RedisCasRace,
    RedisCasException,
    RedisBackendMaxRetriesExceeded,
}

impl ExecutionBranch {
    fn as_str(&self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::NoRoutingId => "no_routing_id",
            Self::OccupiedHit => "occupied_hit",
            Self::OccupiedMiss => "occupied_miss",
            Self::Vacant => "vacant",
            Self::RedisPoolGetException => "redis_pool_get_exception",
            Self::RedisGetexException => "redis_getex_exception",
            Self::RedisCasRace => "redis_cas_race",
            Self::RedisCasException => "redis_cas_exception",
            Self::RedisBackendMaxRetriesExceeded => "redis_backend_max_retries_exceeded",
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
            let (idx, branch) = self
                .backend
                .select_by_routing_id(routing_id, workers, &healthy_indices, self.assignment_mode)
                .await;
            return (
                idx.or_else(|| Some(random_select(&healthy_indices))),
                branch,
            );
        }

        (
            Some(random_select(&healthy_indices)),
            ExecutionBranch::NoRoutingId,
        )
    }

    #[cfg(test)]
    fn local_backend(&self) -> Option<&LocalBackend> {
        self.backend.as_local()
    }
}

#[async_trait]
impl LoadBalancingPolicy for ManualPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
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
    ) -> (Option<usize>, ExecutionBranch) {
        match self {
            Backend::Local(b) => {
                let (idx, branch) =
                    b.select_by_routing_id(routing_id, workers, healthy_indices, assignment_mode);
                (Some(idx), branch)
            }
            Backend::Redis(b) => {
                b.select_by_routing_id(routing_id, workers, healthy_indices, assignment_mode)
                    .await
            }
        }
    }

    fn len(&self) -> Option<usize> {
        match self {
            Backend::Local(b) => Some(b.len()),
            Backend::Redis(_) => None,
        }
    }

    #[cfg(test)]
    fn as_local(&self) -> Option<&LocalBackend> {
        match self {
            Backend::Local(b) => Some(b),
            Backend::Redis(_) => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct CandidateWorkerUrls(Vec<String>);

impl CandidateWorkerUrls {
    fn push_bounded(&mut self, url: String) {
        while self.0.len() >= MAX_CANDIDATE_WORKERS {
            self.0.remove(0);
        }
        self.0.push(url);
    }

    fn serialize(&self) -> String {
        self.0.join(",")
    }

    fn deserialize(data: &str) -> Self {
        Self(
            data.split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect(),
        )
    }

    fn urls(&self) -> &[String] {
        &self.0
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
    candidates: CandidateWorkerUrls,
    last_access: Instant,
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
                if let Some(idx) =
                    find_healthy_worker(node.candidates.urls(), workers, healthy_indices)
                {
                    (idx, ExecutionBranch::OccupiedHit)
                } else {
                    let selected_idx = select_new_worker(workers, healthy_indices, assignment_mode);
                    node.candidates
                        .push_bounded(workers[selected_idx].url().to_string());
                    (selected_idx, ExecutionBranch::OccupiedMiss)
                }
            }
            Entry::Vacant(entry) => {
                let selected_idx = select_new_worker(workers, healthy_indices, assignment_mode);
                let mut candidates = CandidateWorkerUrls::default();
                candidates.push_bounded(workers[selected_idx].url().to_string());
                entry.insert(LocalNode {
                    candidates,
                    last_access: Instant::now(),
                });
                (selected_idx, ExecutionBranch::Vacant)
            }
        }
    }

    fn len(&self) -> usize {
        self.routing_map.len()
    }

    #[cfg(test)]
    fn get_last_access(&self, routing_id: &str) -> Option<Instant> {
        self.routing_map.get(routing_id).map(|e| e.last_access)
    }

    #[cfg(test)]
    fn has_eviction_task(&self) -> bool {
        self._eviction_task.is_some()
    }

    #[cfg(test)]
    fn iter_first_candidate_urls(&self) -> impl Iterator<Item = String> + '_ {
        self.routing_map
            .iter()
            .filter_map(|e| e.candidates.urls().first().cloned())
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

    async fn select_by_routing_id(
        &self,
        routing_id: &str,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        assignment_mode: ManualAssignmentMode,
    ) -> (Option<usize>, ExecutionBranch) {
        let retry_config = RetryConfig {
            max_retries: 5,
            initial_backoff_ms: 10,
            max_backoff_ms: 500,
            backoff_multiplier: 2.0,
            jitter_factor: 0.2,
        };

        let result = RetryExecutor::execute_with_retry(
            &retry_config,
            |_attempt| {
                self.select_one_attempt(routing_id, workers, healthy_indices, assignment_mode)
            },
            |(idx, _branch), _attempt| idx.is_none(),
            |(_idx, branch), _delay, _attempt| {
                Metrics::record_manual_policy_attempt_error(branch.as_str())
            },
            || warn!("Max retries exceeded for routing_id={}", routing_id),
        )
        .await;

        match result {
            Ok((idx, branch)) => (idx, branch),
            Err(MaxRetriesExceeded { .. }) => {
                (None, ExecutionBranch::RedisBackendMaxRetriesExceeded)
            }
        }
    }

    async fn select_one_attempt(
        &self,
        routing_id: &str,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        assignment_mode: ManualAssignmentMode,
    ) -> (Option<usize>, ExecutionBranch) {
        let key = format!("{}{}", REDIS_KEY_PREFIX, routing_id).clone();

        let mut conn = match self.pool.get().await {
            Ok(x) => x,
            Err(e) => {
                warn!("Redis pool.get exception: {}", e);
                return (None, ExecutionBranch::RedisPoolGetException);
            }
        };

        let old_data: Option<String> = match conn.get_ex(&key, Expiry::EX(self.ttl_secs)).await {
            Ok(x) => x,
            Err(e) => {
                warn!("Redis getex exception: {}", e);
                return (None, ExecutionBranch::RedisGetexException);
            }
        };
        let old_candidates = old_data.as_deref().map(CandidateWorkerUrls::deserialize);

        if let Some(ref old_candidates) = old_candidates {
            if let Some(idx) = find_healthy_worker(old_candidates.urls(), workers, healthy_indices)
            {
                return (Some(idx), ExecutionBranch::OccupiedHit);
            }
        }

        let selected_idx = select_new_worker(workers, healthy_indices, assignment_mode);
        let new_url = workers[selected_idx].url();

        let (new_candidates, branch) = if let Some(mut candidates) = old_candidates {
            candidates.push_bounded(new_url.to_string());
            (candidates, ExecutionBranch::OccupiedMiss)
        } else {
            (
                CandidateWorkerUrls(vec![new_url.to_string()]),
                ExecutionBranch::Vacant,
            )
        };
        let new_data = new_candidates.serialize();

        match RedisCommandUtil::cas(
            &mut conn,
            &key,
            old_data.as_deref(),
            &new_data,
            self.ttl_secs,
        )
        .await
        {
            Ok(true) => (Some(selected_idx), branch),
            Ok(false) => (None, ExecutionBranch::RedisCasRace),
            Err(e) => {
                warn!("Redis cas exception: {}", e);
                (None, ExecutionBranch::RedisCasException)
            }
        }
    }
}

// ------------------------------------ redis utils ---------------------------------------

struct RedisCommandUtil;

impl RedisCommandUtil {
    async fn cas(
        conn: &mut deadpool_redis::Connection,
        key: &str,
        expected: Option<&str>,
        new_value: &str,
        ttl_secs: u64,
    ) -> Result<bool, redis::RedisError> {
        static CAS_SCRIPT: std::sync::LazyLock<redis::Script> = std::sync::LazyLock::new(|| {
            redis::Script::new(
                r#"
local old = redis.call('GET', KEYS[1])
local expected = ARGV[1]
local match = (expected == '' and old == false) or (old == expected)
if not match then return 0 end
redis.call('SET', KEYS[1], ARGV[2], 'EX', tonumber(ARGV[3]))
return 1
"#,
            )
        });

        let result: i32 = CAS_SCRIPT
            .key(key)
            .arg(expected.unwrap_or(""))
            .arg(new_value)
            .arg(ttl_secs)
            .invoke_async(conn)
            .await?;
        Ok(result == 1)
    }
}

// ------------------------------------ misc utils ---------------------------------------

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

        let (first_result, branch) = policy.select_worker_impl(&workers, &info).await;
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::Vacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info).await;
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
            let (result, branch) = policy.select_worker_impl(&workers, &info).await;
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
            let (result, branch) = policy.select_worker_impl(&workers, &info).await;
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

        let (result, branch) = policy.select_worker_impl(&workers, &info).await;
        assert_eq!(result, Some(1), "Should only select healthy worker");
        assert_eq!(branch, ExecutionBranch::Vacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info).await;
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
        let (result, branch) = policy.select_worker_impl(&workers, &info).await;
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
            let (result, branch) = policy.select_worker_impl(&workers, &info).await;
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

        let (first_result, branch) = policy.select_worker_impl(&workers, &info).await;
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::Vacant);

        workers[first_idx].set_healthy(false);

        let (new_result, branch) = policy.select_worker_impl(&workers, &info).await;
        let new_idx = new_result.unwrap();
        assert_ne!(new_idx, first_idx, "Should remap to healthy worker");
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info).await;
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
        let (result, branch) = policy.select_worker_impl(&workers, &info).await;
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

        let (result, branch) = policy.select_worker_impl(&workers, &info).await;
        assert_eq!(result, Some(0));
        assert_eq!(branch, ExecutionBranch::Vacant);

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info).await;
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

        let (first_result, branch) = policy.select_worker_impl(&workers, &info).await;
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::Vacant);

        workers[first_idx].set_healthy(false);

        let (second_result, branch) = policy.select_worker_impl(&workers, &info).await;
        let second_idx = second_result.unwrap();
        assert_ne!(second_idx, first_idx);
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        workers[first_idx].set_healthy(true);

        let (after_recovery, branch) = policy.select_worker_impl(&workers, &info).await;
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

        let (first_result, branch) = policy.select_worker_impl(&workers, &info).await;
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::Vacant);

        workers[first_idx].set_healthy(false);

        let (second_result, branch) = policy.select_worker_impl(&workers, &info).await;
        let second_idx = second_result.unwrap();
        assert_ne!(second_idx, first_idx);
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        workers[second_idx].set_healthy(false);

        let remaining_idx = (0..3).find(|&i| i != first_idx && i != second_idx).unwrap();
        let (third_result, branch) = policy.select_worker_impl(&workers, &info).await;
        assert_eq!(
            third_result,
            Some(remaining_idx),
            "Should select the only remaining healthy worker"
        );
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);

        workers[first_idx].set_healthy(true);

        let (idx_after_restore, branch) = policy.select_worker_impl(&workers, &info).await;
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
    async fn test_candidate_worker_urls_push_bounded() {
        let mut candidates = CandidateWorkerUrls::default();
        candidates.push_bounded("http://w1:8000".to_string());

        candidates.push_bounded("http://w2:8000".to_string());
        assert_eq!(candidates.urls().len(), 2);
        assert_eq!(candidates.urls()[0], "http://w1:8000");
        assert_eq!(candidates.urls()[1], "http://w2:8000");

        candidates.push_bounded("http://w3:8000".to_string());
        assert_eq!(candidates.urls().len(), 2);
        assert_eq!(
            candidates.urls()[0],
            "http://w2:8000",
            "Oldest entry should be removed"
        );
        assert_eq!(candidates.urls()[1], "http://w3:8000");
    }

    #[tokio::test]
    async fn test_candidate_worker_urls_serialize_deserialize() {
        let mut candidates = CandidateWorkerUrls::default();
        candidates.push_bounded("http://w1:8000".to_string());
        candidates.push_bounded("http://w2:8000".to_string());

        let serialized = candidates.serialize();
        assert_eq!(serialized, "http://w1:8000,http://w2:8000");

        let deserialized = CandidateWorkerUrls::deserialize(&serialized);
        assert_eq!(deserialized.urls(), candidates.urls());

        let empty = CandidateWorkerUrls::deserialize("");
        assert!(empty.urls().is_empty());
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
        assert!(!policy.local_backend().unwrap().has_eviction_task());
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

        let (result, branch) = policy.select_worker_impl(&workers, &info).await;
        assert_eq!(branch, ExecutionBranch::Vacant);
        let first_idx = result.unwrap();
        let access_after_vacant = policy
            .local_backend()
            .unwrap()
            .get_last_access("test-key")
            .unwrap();
        assert!(access_after_vacant.elapsed().as_millis() < 100);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let (_, branch) = policy.select_worker_impl(&workers, &info).await;
        assert_eq!(branch, ExecutionBranch::OccupiedHit);
        let access_after_hit = policy
            .local_backend()
            .unwrap()
            .get_last_access("test-key")
            .unwrap();
        assert!(access_after_hit > access_after_vacant);

        std::thread::sleep(std::time::Duration::from_millis(10));

        workers[first_idx].set_healthy(false);
        let (_, branch) = policy.select_worker_impl(&workers, &info).await;
        assert_eq!(branch, ExecutionBranch::OccupiedMiss);
        let access_after_miss = policy
            .local_backend()
            .unwrap()
            .get_last_access("test-key")
            .unwrap();
        assert!(access_after_miss > access_after_hit);
    }

    #[tokio::test]
    async fn test_manual_ttl_eviction_logic() {
        use std::time::Duration;

        let config = ManualConfig {
            eviction_interval_secs: 1,
            max_idle_secs: 1,
            assignment_mode: ManualAssignmentMode::Random,
        };
        let policy = ManualPolicy::with_config(config);
        let workers = create_workers(&["http://w1:8000", "http://w2:8000"]);

        let headers = headers_with_routing_key("key-0");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        policy.select_worker_impl(&workers, &info).await;

        assert_eq!(policy.local_backend().unwrap().len(), 1);

        std::thread::sleep(Duration::from_secs(3));

        assert_eq!(policy.local_backend().unwrap().len(), 0);
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

            let (result, branch) = policy.select_worker_impl(&workers, &info).await;
            assert!(result.is_some());
            assert_eq!(branch, ExecutionBranch::Vacant);

            let selected_idx = result.unwrap();
            workers[selected_idx]
                .worker_routing_key_load()
                .increment(&routing_key);
        }

        let distribution: HashMap<_, usize> = policy
            .local_backend()
            .unwrap()
            .iter_first_candidate_urls()
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
        let (result, _) = policy.select_worker_impl(&workers, &info).await;
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
        let (result, _) = policy.select_worker_impl(&workers, &info).await;
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

        let (first_result, branch) = policy.select_worker_impl(&workers, &info).await;
        let first_idx = first_result.unwrap();
        assert_eq!(branch, ExecutionBranch::Vacant);
        assert_eq!(
            first_idx, 0,
            "Should select worker 0 (has 1 routing key vs 2)"
        );

        for _ in 0..10 {
            let (result, branch) = policy.select_worker_impl(&workers, &info).await;
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
            let (result, _) = policy.select_worker_impl(&workers, &info).await;
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
