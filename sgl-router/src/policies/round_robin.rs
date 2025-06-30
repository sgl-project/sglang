use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use async_trait::async_trait;

use crate::worker::Worker;

use super::{LoadBalancing, RoutingError, RoutingPolicy};

pub struct RoundRobinPolicy {
    single_counter: AtomicUsize,
    prefill_counter: AtomicUsize,
    decode_counter: AtomicUsize,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            single_counter: AtomicUsize::new(0),
            prefill_counter: AtomicUsize::new(0),
            decode_counter: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl RoutingPolicy for RoundRobinPolicy {
    async fn select_single(
        &self,
        workers: &[Arc<dyn Worker>],
        _request: &serde_json::Value,
    ) -> Result<Arc<dyn Worker>, RoutingError> {
        let healthy_workers = self.get_healthy_workers(workers);
        if healthy_workers.is_empty() {
            return Err(RoutingError::NoHealthyWorkers);
        }

        let index = self.single_counter.fetch_add(1, Ordering::Relaxed);
        Ok(healthy_workers[index % healthy_workers.len()].clone())
    }

    async fn select_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        _request: &serde_json::Value,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), RoutingError> {
        let healthy_prefill = self.get_healthy_workers(prefill_workers);
        if healthy_prefill.is_empty() {
            return Err(RoutingError::NoHealthyWorkers);
        }
        let index = self.prefill_counter.fetch_add(1, Ordering::Relaxed);
        let prefill_worker = healthy_prefill[index % healthy_prefill.len()].clone();

        let healthy_decode = self.get_healthy_workers(decode_workers);
        if healthy_decode.is_empty() {
            return Err(RoutingError::NoHealthyWorkers);
        }
        let index = self.decode_counter.fetch_add(1, Ordering::Relaxed);
        let decode_worker = healthy_decode[index % healthy_decode.len()].clone();

        Ok((prefill_worker, decode_worker))
    }

    fn on_request_complete(&self, _: &str, _: bool) {}

    fn name(&self) -> &'static str {
        "round_robin"
    }

    // fn add_worker(&self, _worker: Arc<dyn Worker>) {}

    // fn remove_worker(&self, _worker_url: &str) {}
}

impl LoadBalancing for RoundRobinPolicy {}
