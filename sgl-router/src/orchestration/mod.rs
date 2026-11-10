//! Adapter from the Router's worker registry to session KV orchestration.
use std::sync::Arc;

use reqwest::header::HeaderValue;
pub use sgl_kv_orchestrator::{KvContext, OrchestrationConfig, Runtime};

use crate::{
    discovery::{ModelId, WorkerMode},
    workers::WorkerRegistry,
};

pub(crate) fn start(
    config: OrchestrationConfig,
    registry: Arc<WorkerRegistry>,
    model: ModelId,
    auth: Option<HeaderValue>,
) -> Arc<Runtime> {
    Runtime::start(
        config,
        move || {
            let mut workers = registry.healthy_workers_for(&model);
            workers.retain(|w| matches!(w.mode(), WorkerMode::Plain | WorkerMode::Prefill));
            workers.sort_by(|a, b| {
                a.router_inflight_load()
                    .cmp(&b.router_inflight_load())
                    .then_with(|| a.id.0.cmp(&b.id.0))
            });
            workers.into_iter().map(|w| w.url.clone()).collect()
        },
        auth,
    )
}
