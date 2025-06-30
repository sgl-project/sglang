use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use async_trait::async_trait;

use crate::worker::Worker;

use super::error::RoutingError;

#[async_trait]
pub trait RoutingPolicy: Send + Sync {
    /// Select a single worker for regular routing
    async fn select_single(
        &self,
        workers: &[Arc<dyn Worker>],
        request: &serde_json::Value,
    ) -> Result<Arc<dyn Worker>, RoutingError>;

    /// Select prefill and decode workers for PD routing
    async fn select_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request: &serde_json::Value,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), RoutingError>;

    /// Notify policy of request completion (for stateful policies)
    fn on_request_complete(&self, worker_url: &str, success: bool);

    /// Get policy name for metrics and debugging
    fn name(&self) -> &'static str;

    // Add a worker to the policy's state
    // fn add_worker(&self, worker: Arc<dyn Worker>); // todo(Yingyi): not in #7535, might be added

    // Remove a worker from the policy's state
    // fn remove_worker(&self, worker_url: &str); // todo(Yingyi): not in #7535, might be added
}

pub trait LoadBalancing {
    fn select_least_loaded(&self, workers: &[Arc<dyn Worker>]) -> Option<Arc<dyn Worker>> {
        workers
            .iter()
            .filter(|w| w.is_healthy())
            .min_by_key(|w| w.load().load(Ordering::Relaxed))
            .cloned()
    }

    fn get_healthy_workers(&self, workers: &[Arc<dyn Worker>]) -> Vec<Arc<dyn Worker>> {
        workers.iter().filter(|w| w.is_healthy()).cloned().collect()
    }
}
