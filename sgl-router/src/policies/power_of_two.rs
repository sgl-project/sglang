use std::sync::atomic::Ordering;
use std::sync::Arc;

use async_trait::async_trait;
use rand::seq::SliceRandom;

use crate::config::{PolicyConfig, PolicyError};
use crate::worker::Worker;

use super::error::RoutingError;
use super::traits::{LoadBalancing, RoutingPolicy};

pub struct PowerOfTwoPolicy;

impl PowerOfTwoPolicy {
    pub fn new(_config: &PolicyConfig, _workers: &[Arc<dyn Worker>]) -> Result<Self, PolicyError> {
        Ok(Self)
    }
}

#[async_trait]
impl RoutingPolicy for PowerOfTwoPolicy {
    async fn select_single(
        &self,
        workers: &[Arc<dyn Worker>],
        _request: &serde_json::Value,
    ) -> Result<Arc<dyn Worker>, RoutingError> {
        let healthy_workers = self.get_healthy_workers(workers);

        match healthy_workers.len() {
            0 => Err(RoutingError::NoHealthyWorkers),
            1 => Ok(healthy_workers[0].clone()),
            _ => {
                let mut rng = rand::thread_rng();
                let chosen = healthy_workers
                    .choose_multiple(&mut rng, 2)
                    .cloned()
                    .collect::<Vec<_>>();
                let worker1 = &chosen[0];
                let worker2 = &chosen[1];

                if worker1.load().load(Ordering::Relaxed) <= worker2.load().load(Ordering::Relaxed)
                {
                    Ok(worker1.clone())
                } else {
                    Ok(worker2.clone())
                }
            }
        }
    }

    async fn select_pair(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request: &serde_json::Value,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), RoutingError> {
        let p = self.select_single(prefill_workers, request).await?;
        let d = self.select_single(decode_workers, request).await?;
        Ok((p, d))
    }

    fn on_request_complete(&self, _worker_url: &str, _success: bool) {}

    fn name(&self) -> &'static str {
        "power_of_two"
    }

    // fn add_worker(&self, _worker: Arc<dyn Worker>) {}

    // fn remove_worker(&self, _worker_url: &str) {}
}

impl LoadBalancing for PowerOfTwoPolicy {}
