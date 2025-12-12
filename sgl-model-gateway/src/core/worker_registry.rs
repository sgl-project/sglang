//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing

use std::sync::{Arc, RwLock};

use dashmap::DashMap;
use uuid::Uuid;

use crate::{
    core::{ConnectionMode, RuntimeType, Worker, WorkerType},
    observability::metrics::RouterMetrics,
};

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

/// Model index type for O(1) lookups (stores Arc<dyn Worker> directly)
type ModelIndex = Arc<DashMap<String, Arc<RwLock<Vec<Arc<dyn Worker>>>>>>;

/// Worker registry with model-based indexing
#[derive(Debug)]
pub struct WorkerRegistry {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Model index for O(1) lookups (stores Arc<dyn Worker> directly)
    /// This replaces the previous dual-index approach for better memory efficiency
    model_index: ModelIndex,

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
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
        }
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

        // Update model index for O(1) lookups
        let model_id = worker.model_id().to_string();
        self.model_index
            .entry(model_id)
            .or_insert_with(|| Arc::new(RwLock::new(Vec::new())))
            .write()
            .expect("RwLock for model_index is poisoned")
            .push(worker.clone());

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

    /// Remove a worker by ID
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker)) = self.workers.remove(worker_id) {
            // Remove from URL mapping
            self.url_to_id.remove(worker.url());

            // Remove from model index
            if let Some(model_index_entry) = self.model_index.get(worker.model_id()) {
                let worker_url = worker.url();
                model_index_entry
                    .write()
                    .expect("RwLock for model_index is poisoned")
                    .retain(|w| w.url() != worker_url);
            }

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
            RouterMetrics::remove_worker_metrics(worker.url());

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

    /// Get all workers for a model (O(1) optimized)
    /// Uses the pre-indexed model_index for fast lookups
    pub fn get_by_model(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        self.model_index
            .get(model_id)
            .map(|workers| {
                workers
                    .read()
                    .expect("RwLock for model_index is poisoned")
                    .clone()
            })
            .unwrap_or_default()
    }

    /// Alias for get_by_model for backwards compatibility
    #[inline]
    pub fn get_by_model_fast(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        self.get_by_model(model_id)
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

    /// Get all model IDs with workers
    pub fn get_models(&self) -> Vec<String> {
        self.model_index
            .iter()
            .filter(|entry| {
                entry
                    .value()
                    .read()
                    .map(|workers| !workers.is_empty())
                    .unwrap_or(false)
            })
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
        let workers = if let Some(model) = model_id {
            self.get_by_model_fast(model)
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

    /// Get worker statistics
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        // Count models directly instead of allocating Vec via get_models()
        let total_models = self
            .model_index
            .iter()
            .filter(|entry| {
                entry
                    .value()
                    .read()
                    .map(|workers| !workers.is_empty())
                    .unwrap_or(false)
            })
            .count();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;

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
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            total_load,
            regular_workers: regular_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
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
    pub fn start_health_checker(&self, check_interval_secs: u64) -> crate::core::HealthChecker {
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

            // Counter for periodic load reset (every 10 health check cycles)
            let mut check_count = 0u64;
            const LOAD_RESET_INTERVAL: u64 = 10;

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

                // Reset loads periodically
                check_count += 1;
                if check_count.is_multiple_of(LOAD_RESET_INTERVAL) {
                    tracing::debug!("Resetting worker loads (cycle {})", check_count);
                    for worker in &workers {
                        worker.reset_load();
                    }
                }
            }
        });

        crate::core::HealthChecker::new(handle, shutdown)
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
    pub total_workers: usize,
    pub total_models: usize,
    pub healthy_workers: usize,
    pub total_load: usize,
    pub regular_workers: usize,
    pub prefill_workers: usize,
    pub decode_workers: usize,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{BasicWorkerBuilder, CircuitBreakerConfig};

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

        // Register worker (WorkerFactory returns Box<dyn Worker>, convert to Arc)
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

        let llama_workers = registry.get_by_model_fast("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        let gpt_workers = registry.get_by_model_fast("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        let unknown_workers = registry.get_by_model_fast("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        let llama_workers_slow = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), llama_workers_slow.len());

        registry.remove_by_url("http://worker1:8080");
        let llama_workers_after = registry.get_by_model_fast("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }
}
