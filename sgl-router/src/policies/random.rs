use std::sync::Arc;

use async_trait::async_trait;
use rand::Rng;

use crate::worker::Worker;

use super::{LoadBalancing, RoutingError, RoutingPolicy};

pub struct RandomPolicy;

impl RandomPolicy {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl RoutingPolicy for RandomPolicy {
    async fn select_single(
        &self,
        workers: &[Arc<dyn Worker>],
        _request: &serde_json::Value,
    ) -> Result<Arc<dyn Worker>, RoutingError> {
        let healthy_workers = self.get_healthy_workers(workers);

        if healthy_workers.is_empty() {
            return Err(RoutingError::NoHealthyWorkers);
        }

        let idx = rand::thread_rng().gen_range(0..healthy_workers.len());
        Ok(healthy_workers[idx].clone())
    }

    async fn select_pair(
        &self,
        prefill: &[Arc<dyn Worker>],
        decode: &[Arc<dyn Worker>],
        request: &serde_json::Value,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), RoutingError> {
        let p = self.select_single(prefill, request).await?;
        let d = self.select_single(decode, request).await?;
        Ok((p, d))
    }

    fn on_request_complete(&self, _: &str, _: bool) {}

    fn name(&self) -> &'static str {
        "random"
    }

    // fn add_worker(&self, _worker: Arc<dyn Worker>) {}

    // fn remove_worker(&self, _worker_url: &str) {}
}

impl LoadBalancing for RandomPolicy {}
