//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing

use crate::core::{ConnectionMode, Worker, WorkerType};
use dashmap::DashMap;
use std::sync::Arc;
use uuid::Uuid;

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

/// Worker registry with model-based indexing
pub struct WorkerRegistry {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Workers indexed by model ID
    model_workers: Arc<DashMap<String, Vec<WorkerId>>>,

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping (for backward compatibility)
    url_to_id: Arc<DashMap<String, WorkerId>>,
}

impl WorkerRegistry {
    /// Create a new worker registry
    pub fn new() -> Self {
        Self {
            workers: Arc::new(DashMap::new()),
            model_workers: Arc::new(DashMap::new()),
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

        // Update model index
        let model_id = worker.model_id().to_string();
        self.model_workers
            .entry(model_id)
            .or_default()
            .push(worker_id.clone());

        // Update type index
        self.type_workers
            .entry(worker.worker_type())
            .or_default()
            .push(worker_id.clone());

        // Update connection mode index
        self.connection_workers
            .entry(worker.connection_mode())
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
            if let Some(mut model_workers) = self.model_workers.get_mut(worker.model_id()) {
                model_workers.retain(|id| id != worker_id);
            }

            // Remove from type index
            if let Some(mut type_workers) = self.type_workers.get_mut(&worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }

            // Remove from connection mode index
            if let Some(mut conn_workers) =
                self.connection_workers.get_mut(&worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }

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
        self.url_to_id.get(url).and_then(|id| self.get(&id.clone()))
    }

    /// Get all workers for a model
    pub fn get_by_model(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        self.model_workers
            .get(model_id)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all workers by worker type
    pub fn get_by_type(&self, worker_type: &WorkerType) -> Vec<Arc<dyn Worker>> {
        self.type_workers
            .get(worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
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

    /// Get all worker URLs
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    /// Get all model IDs with workers
    pub fn get_models(&self) -> Vec<String> {
        self.model_workers
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get worker statistics
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        let total_models = self.get_models().len();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;

        for worker in self.get_all() {
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
    use super::*;
    use crate::core::{CircuitBreakerConfig, WorkerFactory};
    use std::collections::HashMap;

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        // Create a worker with labels
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker = WorkerFactory::create_regular_with_labels(
            "http://worker1:8080".to_string(),
            labels,
            CircuitBreakerConfig::default(),
        );

        // Register worker (WorkerFactory returns Box<dyn Worker>, convert to Arc)
        let worker_id = registry.register(Arc::from(worker));

        // Verify registration
        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);

        // Test stats
        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        // Remove worker
        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }
}
