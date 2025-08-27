use super::{BasicWorker, Worker, WorkerType};
use crate::core::worker::WorkerMetadata;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

/// SGLang-specific worker that can fetch capacity information from the backend
#[derive(Debug, Clone)]
pub struct SGLangWorker {
    // Delegate basic functionality to BasicWorker
    base: BasicWorker,
    // Cached capacity value
    cached_capacity: Arc<AtomicUsize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ServerInfo {
    /// The configured maximum concurrent requests this server can handle.
    /// This is NOT the current number of running requests, but rather the capacity limit
    /// set via --max-running-requests when the server was started.
    max_running_requests: Option<usize>,
    #[serde(flatten)]
    other: serde_json::Value,
}

impl SGLangWorker {
    /// Create a new SGLang worker
    pub fn new(url: String, worker_type: WorkerType) -> Self {
        Self {
            base: BasicWorker::new(url, worker_type),
            cached_capacity: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Fetch capacity information from the SGLang backend
    async fn fetch_capacity(&self) -> Option<usize> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .ok()?;

        let url = format!("{}/get_server_info", self.base.url());

        match client.get(&url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<ServerInfo>().await {
                        Ok(info) => {
                            if let Some(capacity) = info.max_running_requests {
                                debug!(
                                    "Worker {} reported capacity: {}",
                                    self.base.url(),
                                    capacity
                                );
                                self.cached_capacity.store(capacity, Ordering::Relaxed);
                                Some(capacity)
                            } else {
                                debug!(
                                    "Worker {} did not report max_running_requests",
                                    self.base.url()
                                );
                                None
                            }
                        }
                        Err(e) => {
                            warn!(
                                "Failed to parse server info from {}: {}",
                                self.base.url(),
                                e
                            );
                            None
                        }
                    }
                } else {
                    warn!(
                        "Server info request to {} returned status: {}",
                        self.base.url(),
                        response.status()
                    );
                    None
                }
            }
            Err(e) => {
                debug!(
                    "Failed to fetch server info from {}: {}",
                    self.base.url(),
                    e
                );
                None
            }
        }
    }
}

#[async_trait]
impl Worker for SGLangWorker {
    fn url(&self) -> &str {
        self.base.url()
    }

    fn worker_type(&self) -> WorkerType {
        self.base.worker_type()
    }

    fn is_healthy(&self) -> bool {
        self.base.is_healthy()
    }

    fn set_healthy(&self, healthy: bool) {
        self.base.set_healthy(healthy)
    }

    async fn check_health_async(&self) -> super::WorkerResult<()> {
        // During health check, also try to update capacity
        let health_result = self.base.check_health_async().await;

        if health_result.is_ok() {
            // Try to fetch capacity in the background
            let self_clone = self.clone();
            tokio::spawn(async move {
                self_clone.fetch_capacity().await;
            });
        }

        health_result
    }

    fn load(&self) -> usize {
        self.base.load()
    }

    fn capacity(&self) -> Option<usize> {
        let capacity = self.cached_capacity.load(Ordering::Relaxed);
        if capacity > 0 {
            Some(capacity)
        } else {
            None
        }
    }

    fn increment_load(&self) {
        self.base.increment_load()
    }

    fn decrement_load(&self) {
        self.base.decrement_load()
    }

    fn reset_load(&self) {
        self.base.reset_load()
    }

    fn processed_requests(&self) -> usize {
        self.base.processed_requests()
    }

    fn increment_processed(&self) {
        self.base.increment_processed()
    }

    fn metadata(&self) -> &WorkerMetadata {
        self.base.metadata()
    }

    fn clone_worker(&self) -> Box<dyn Worker> {
        Box::new(self.clone())
    }

    fn circuit_breaker(&self) -> &super::CircuitBreaker {
        self.base.circuit_breaker()
    }
}

/// Factory function to create SGLang workers
pub fn create_sglang_worker(url: String, worker_type: WorkerType) -> Arc<dyn Worker> {
    Arc::new(SGLangWorker::new(url, worker_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sglang_worker_creation() {
        let worker = SGLangWorker::new("http://test:8080".to_string(), WorkerType::Regular);
        assert_eq!(worker.url(), "http://test:8080");
        assert_eq!(worker.worker_type(), WorkerType::Regular);
        assert!(worker.capacity().is_none()); // No capacity until fetched
    }

    #[tokio::test]
    async fn test_cached_capacity() {
        let worker = SGLangWorker::new("http://test:8080".to_string(), WorkerType::Regular);

        // Manually set cached capacity
        worker.cached_capacity.store(100, Ordering::Relaxed);

        assert_eq!(worker.capacity(), Some(100));
    }
}
