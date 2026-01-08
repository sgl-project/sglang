//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing
//!
//! # Performance Optimizations
//! The model index uses immutable Arc snapshots instead of RwLock for lock-free reads.
//! This is critical for high-concurrency scenarios where many requests query the same model.
//!
//! # Consistent Hash Ring
//! The registry maintains a pre-computed hash ring per model for O(log n) consistent hashing.
//! The ring is rebuilt only when workers are added/removed, not per-request.
//! Uses virtual nodes (150 per worker) for even distribution and blake3 for stable hashing.

use std::sync::Arc;

use dashmap::DashMap;
use uuid::Uuid;

use crate::{
    core::{
        circuit_breaker::CircuitState,
        worker::{HealthChecker, RuntimeType, WorkerType},
        ConnectionMode, Worker,
    },
    observability::metrics::Metrics,
};

/// Number of virtual nodes per physical worker for even distribution.
/// 150 is a common choice that provides good balance between memory and distribution.
const VIRTUAL_NODES_PER_WORKER: usize = 150;

/// Consistent hash ring for O(log n) worker selection.
///
/// Each worker is placed at multiple positions (virtual nodes) on the ring
/// based on hash(worker_url + vnode_index). This provides:
/// - Even key distribution across workers
/// - Minimal key redistribution when workers are added/removed (~1/N keys move)
/// - O(log n) lookup via binary search
///
/// Uses blake3 for stable, fast hashing that's consistent across Rust versions.
#[derive(Debug, Clone)]
pub struct HashRing {
    /// Sorted list of (ring_position, worker_url)
    /// Multiple entries per worker (virtual nodes) for even distribution.
    /// Uses Arc<str> to share URL across all virtual nodes (150 refs vs 150 copies).
    entries: Arc<[(u64, Arc<str>)]>,
}

impl HashRing {
    /// Build a hash ring from a list of workers.
    /// Creates VIRTUAL_NODES_PER_WORKER entries per worker for even distribution.
    pub fn new(workers: &[Arc<dyn Worker>]) -> Self {
        let mut entries: Vec<(u64, Arc<str>)> =
            Vec::with_capacity(workers.len() * VIRTUAL_NODES_PER_WORKER);

        for worker in workers {
            // Create Arc<str> once per worker, share across all virtual nodes
            let url: Arc<str> = Arc::from(worker.url());

            // Create multiple virtual nodes per worker
            for vnode in 0..VIRTUAL_NODES_PER_WORKER {
                let vnode_key = format!("{}#{}", url, vnode);
                let pos = Self::hash_position(&vnode_key);
                entries.push((pos, Arc::clone(&url)));
            }
        }

        // Sort by ring position for binary search
        entries.sort_unstable_by_key(|(pos, _)| *pos);

        Self {
            entries: Arc::from(entries.into_boxed_slice()),
        }
    }

    /// Hash a string to a ring position using blake3 (stable across versions).
    #[inline]
    fn hash_position(s: &str) -> u64 {
        let hash = blake3::hash(s.as_bytes());
        // Take first 8 bytes as u64
        u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
    }

    /// Find worker URL for a key using consistent hashing.
    /// Returns the first healthy worker URL at or after the key's position (clockwise).
    ///
    /// - `key`: The routing key to hash
    /// - `is_healthy`: Function to check if a worker URL is healthy
    pub fn find_healthy_url<F>(&self, key: &str, is_healthy: F) -> Option<&str>
    where
        F: Fn(&str) -> bool,
    {
        if self.entries.is_empty() {
            return None;
        }

        let key_pos = Self::hash_position(key);

        // Binary search to find first entry at or after key_pos
        let start = self.entries.partition_point(|(pos, _)| *pos < key_pos);

        // Walk clockwise from start, wrapping around
        // Track visited URLs to avoid checking same worker multiple times (virtual nodes)
        let mut checked_urls =
            std::collections::HashSet::with_capacity(self.worker_count().min(16));

        for i in 0..self.entries.len() {
            let (_, url) = &self.entries[(start + i) % self.entries.len()];
            let url_str: &str = url;

            // Skip if we already checked this worker (from another virtual node)
            if !checked_urls.insert(url_str) {
                continue;
            }

            if is_healthy(url_str) {
                return Some(url_str);
            }
        }

        None
    }

    /// Check if the ring is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries in the ring (including virtual nodes)
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of unique workers in the ring
    pub fn worker_count(&self) -> usize {
        self.entries.len() / VIRTUAL_NODES_PER_WORKER.max(1)
    }
}

/// Unique identifier for a worker
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkerId(String);

impl WorkerId {
    /// Create a new worker ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a worker ID from a string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Model index using immutable snapshots for lock-free reads.
/// Each model maps to an Arc'd slice of workers that can be read without locking.
/// Updates create new snapshots (copy-on-write semantics).
type ModelIndex = Arc<DashMap<String, Arc<[Arc<dyn Worker>]>>>;

/// Worker registry with model-based indexing
#[derive(Debug)]
pub struct WorkerRegistry {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Model index for O(1) lookups using immutable snapshots.
    /// Uses Arc<[T]> instead of Arc<RwLock<Vec<T>>> for lock-free reads.
    model_index: ModelIndex,

    /// Consistent hash rings per model for O(log n) routing.
    /// Rebuilt on worker add/remove (copy-on-write).
    hash_rings: Arc<DashMap<String, Arc<HashRing>>>,

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping
    url_to_id: Arc<DashMap<String, WorkerId>>,
}

impl WorkerRegistry {
    /// Create a new worker registry
    pub fn new() -> Self {
        Self {
            workers: Arc::new(DashMap::new()),
            model_index: Arc::new(DashMap::new()),
            hash_rings: Arc::new(DashMap::new()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
        }
    }

    /// Rebuild the hash ring for a model based on current workers in the model index
    fn rebuild_hash_ring(&self, model_id: &str) {
        if let Some(workers) = self.model_index.get(model_id) {
            let ring = HashRing::new(&workers);
            self.hash_rings.insert(model_id.to_string(), Arc::new(ring));
        } else {
            // No workers for this model, remove the ring
            self.hash_rings.remove(model_id);
        }
    }

    /// Get the hash ring for a model (O(1) lookup)
    pub fn get_hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>> {
        self.hash_rings.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Register a new worker
    pub fn register(&self, worker: Arc<dyn Worker>) -> WorkerId {
        let worker_id = if let Some(existing_id) = self.url_to_id.get(worker.url()) {
            // Worker with this URL already exists, update it
            existing_id.clone()
        } else {
            WorkerId::new()
        };

        // Store worker
        self.workers.insert(worker_id.clone(), worker.clone());

        // Update URL mapping
        self.url_to_id
            .insert(worker.url().to_string(), worker_id.clone());

        // Update model index for O(1) lookups using copy-on-write
        // This creates a new immutable snapshot with the added worker
        let model_id = worker.model_id().to_string();
        self.model_index
            .entry(model_id.clone())
            .and_modify(|existing| {
                // Create new snapshot with the additional worker
                let mut new_workers: Vec<Arc<dyn Worker>> = existing.iter().cloned().collect();
                new_workers.push(worker.clone());
                *existing = Arc::from(new_workers.into_boxed_slice());
            })
            .or_insert_with(|| Arc::from(vec![worker.clone()].into_boxed_slice()));

        // Rebuild hash ring for this model
        self.rebuild_hash_ring(&model_id);

        // Update type index (clone needed for DashMap key ownership)
        self.type_workers
            .entry(worker.worker_type().clone())
            .or_default()
            .push(worker_id.clone());

        // Update connection mode index (clone needed for DashMap key ownership)
        self.connection_workers
            .entry(worker.connection_mode().clone())
            .or_default()
            .push(worker_id.clone());

        worker_id
    }

    /// Reserve (or retrieve) a stable UUID for a worker URL.
    /// Uses atomic entry API to avoid race conditions between check and insert.
    pub fn reserve_id_for_url(&self, url: &str) -> WorkerId {
        self.url_to_id.entry(url.to_string()).or_default().clone()
    }

    /// Best-effort lookup of the URL for a given worker ID.
    pub fn get_url_by_id(&self, worker_id: &WorkerId) -> Option<String> {
        if let Some(worker) = self.get(worker_id) {
            return Some(worker.url().to_string());
        }
        self.url_to_id
            .iter()
            .find_map(|entry| (entry.value() == worker_id).then(|| entry.key().clone()))
    }

    /// Remove a worker by ID
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker)) = self.workers.remove(worker_id) {
            // Remove from URL mapping
            self.url_to_id.remove(worker.url());

            // Remove from model index using copy-on-write
            // Create new snapshot without the removed worker
            let worker_url = worker.url();
            let model_id = worker.model_id().to_string();
            if let Some(mut entry) = self.model_index.get_mut(&model_id) {
                let new_workers: Vec<Arc<dyn Worker>> = entry
                    .iter()
                    .filter(|w| w.url() != worker_url)
                    .cloned()
                    .collect();
                *entry = Arc::from(new_workers.into_boxed_slice());
            }

            // Rebuild hash ring for this model
            self.rebuild_hash_ring(&model_id);

            // Remove from type index
            if let Some(mut type_workers) = self.type_workers.get_mut(worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }

            // Remove from connection mode index
            if let Some(mut conn_workers) =
                self.connection_workers.get_mut(worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }

            worker.set_healthy(false);
            Metrics::remove_worker_metrics(worker.url());

            Some(worker)
        } else {
            None
        }
    }

    /// Remove a worker by URL
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker_id)) = self.url_to_id.remove(url) {
            self.remove(&worker_id)
        } else {
            None
        }
    }

    /// Get a worker by ID
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Get a worker by URL
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.url_to_id.get(url).and_then(|id| self.get(&id))
    }

    /// Empty worker slice constant for returning when no workers found
    const EMPTY_WORKERS: &'static [Arc<dyn Worker>] = &[];

    /// Get all workers for a model (O(1) optimized, lock-free)
    /// Returns an Arc to the immutable worker slice - just an atomic refcount bump.
    /// This is the fastest possible read path with zero contention.
    pub fn get_by_model(&self, model_id: &str) -> Arc<[Arc<dyn Worker>]> {
        self.model_index
            .get(model_id)
            .map(|workers| Arc::clone(&workers))
            .unwrap_or_else(|| Arc::from(Self::EMPTY_WORKERS))
    }

    /// Get all workers by worker type
    pub fn get_by_type(&self, worker_type: &WorkerType) -> Vec<Arc<dyn Worker>> {
        self.type_workers
            .get(worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all prefill workers (regardless of bootstrap_port)
    pub fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .filter_map(|entry| {
                let worker = entry.value();
                match worker.worker_type() {
                    WorkerType::Prefill { .. } => Some(worker.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    /// Get all decode workers
    pub fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.get_by_type(&WorkerType::Decode)
    }

    /// Get all workers by connection mode
    pub fn get_by_connection(&self, connection_mode: &ConnectionMode) -> Vec<Arc<dyn Worker>> {
        self.connection_workers
            .get(connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get the number of workers in the registry
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Get all workers
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all workers with their IDs
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get all worker URLs
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    pub fn get_all_urls_with_api_key(&self) -> Vec<(String, Option<String>)> {
        self.workers
            .iter()
            .map(|entry| {
                (
                    entry.value().url().to_string(),
                    entry.value().api_key().clone(),
                )
            })
            .collect()
    }

    /// Get all model IDs with workers (lock-free)
    pub fn get_models(&self) -> Vec<String> {
        self.model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get workers filtered by multiple criteria
    ///
    /// This method allows flexible filtering of workers based on:
    /// - model_id: Filter by specific model
    /// - worker_type: Filter by worker type (Regular, Prefill, Decode)
    /// - connection_mode: Filter by connection mode (Http, Grpc)
    /// - runtime_type: Filter by runtime type (Sglang, Vllm, External)
    /// - healthy_only: Only return healthy workers
    pub fn get_workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
        runtime_type: Option<RuntimeType>,
        healthy_only: bool,
    ) -> Vec<Arc<dyn Worker>> {
        // Start with the most efficient collection based on filters
        // Use model index when possible as it's O(1) lookup
        let workers: Vec<Arc<dyn Worker>> = if let Some(model) = model_id {
            self.get_by_model(model).to_vec()
        } else {
            self.get_all()
        };

        // Apply remaining filters
        workers
            .into_iter()
            .filter(|w| {
                // Check worker_type if specified
                if let Some(ref wtype) = worker_type {
                    if *w.worker_type() != *wtype {
                        return false;
                    }
                }

                // Check connection_mode if specified (using matches for flexible gRPC matching)
                if let Some(ref conn) = connection_mode {
                    if !w.connection_mode().matches(conn) {
                        return false;
                    }
                }

                // Check runtime_type if specified
                if let Some(ref rt) = runtime_type {
                    if w.metadata().runtime_type != *rt {
                        return false;
                    }
                }

                // Check health if required
                if healthy_only && !w.is_healthy() {
                    return false;
                }

                true
            })
            .collect()
    }

    /// Get worker statistics (lock-free)
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        // Count models directly instead of allocating Vec via get_models() (lock-free)
        let total_models = self
            .model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .count();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;
        let mut http_count = 0;
        let mut grpc_count = 0;
        let mut cb_open_count = 0;
        let mut cb_half_open_count = 0;

        // Iterate DashMap directly to avoid cloning all workers via get_all()
        for entry in self.workers.iter() {
            let worker = entry.value();
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += worker.load();

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Prefill { .. } => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
            }

            match worker.connection_mode() {
                ConnectionMode::Http => http_count += 1,
                ConnectionMode::Grpc { .. } => grpc_count += 1,
            }

            match worker.circuit_breaker().state() {
                CircuitState::Open => cb_open_count += 1,
                CircuitState::HalfOpen => cb_half_open_count += 1,
                CircuitState::Closed => {}
            }
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            unhealthy_workers: total_workers.saturating_sub(healthy_count),
            total_load,
            regular_workers: regular_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
            http_workers: http_count,
            grpc_workers: grpc_count,
            circuit_breaker_open: cb_open_count,
            circuit_breaker_half_open: cb_half_open_count,
        }
    }

    /// Get counts of regular and PD workers efficiently (O(1))
    /// This avoids the overhead of get_all() which allocates memory and iterates all workers
    pub fn get_worker_distribution(&self) -> (usize, usize) {
        // Use the existing type_workers index for O(1) lookup
        let regular_count = self
            .type_workers
            .get(&WorkerType::Regular)
            .map(|v| v.len())
            .unwrap_or(0);

        // Get total workers count efficiently from DashMap
        let total_workers = self.workers.len();

        // PD workers are any workers that are not Regular
        let pd_count = total_workers.saturating_sub(regular_count);

        (regular_count, pd_count)
    }

    /// Start a health checker for all workers in the registry
    /// This should be called once after the registry is populated with workers
    pub(crate) fn start_health_checker(&self, check_interval_secs: u64) -> HealthChecker {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let workers_ref = self.workers.clone();

        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(check_interval_secs));

            loop {
                interval.tick().await;

                // Check for shutdown signal
                if shutdown_clone.load(Ordering::Acquire) {
                    tracing::debug!("Registry health checker shutting down");
                    break;
                }

                // Get all workers from registry
                let workers: Vec<Arc<dyn Worker>> = workers_ref
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // Perform health checks in parallel for better performance
                // This is especially important when there are many workers
                let health_futures: Vec<_> = workers
                    .iter()
                    .map(|worker| {
                        let worker = worker.clone();
                        async move {
                            let _ = worker.check_health_async().await;
                        }
                    })
                    .collect();
                futures::future::join_all(health_futures).await;
            }
        });

        HealthChecker::new(handle, shutdown)
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the worker registry
#[derive(Debug, Clone)]
pub struct WorkerRegistryStats {
    /// Total number of registered workers
    pub total_workers: usize,
    /// Number of unique models served
    pub total_models: usize,
    /// Number of workers passing health checks
    pub healthy_workers: usize,
    /// Number of workers failing health checks
    pub unhealthy_workers: usize,
    /// Sum of current load across all workers
    pub total_load: usize,
    /// Number of regular (non-PD) workers
    pub regular_workers: usize,
    /// Number of prefill workers (PD mode)
    pub prefill_workers: usize,
    /// Number of decode workers (PD mode)
    pub decode_workers: usize,
    /// Number of HTTP-connected workers
    pub http_workers: usize,
    /// Number of gRPC-connected workers
    pub grpc_workers: usize,
    /// Number of workers with circuit breaker in Open state (not accepting requests)
    pub circuit_breaker_open: usize,
    /// Number of workers with circuit breaker in HalfOpen state (testing recovery)
    pub circuit_breaker_half_open: usize,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{circuit_breaker::CircuitBreakerConfig, BasicWorkerBuilder};

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        // Create a worker with labels
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register worker
        let worker_id = registry.register(Arc::from(worker));

        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);

        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        // Remove worker
        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        // Create workers for different models
        let mut labels1 = HashMap::new();
        labels1.insert("model_id".to_string(), "llama-3".to_string());
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels1)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels2 = HashMap::new();
        labels2.insert("model_id".to_string(), "llama-3".to_string());
        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels2)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels3 = HashMap::new();
        labels3.insert("model_id".to_string(), "gpt-4".to_string());
        let worker3: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker3:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels3)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register workers
        registry.register(Arc::from(worker1));
        registry.register(Arc::from(worker2));
        registry.register(Arc::from(worker3));

        let llama_workers = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        let gpt_workers = registry.get_by_model("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        let unknown_workers = registry.get_by_model("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        registry.remove_by_url("http://worker1:8080");
        let llama_workers_after = registry.get_by_model("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }
}
