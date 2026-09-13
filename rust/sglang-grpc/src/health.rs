//! Standard gRPC health checks backed by the same runtime state as HealthCheck.

use std::sync::Arc;

use tokio::sync::watch;
use tonic::server::NamedService;
use tonic::{Request, Response, Status};
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_server::Health;
use tonic_health::pb::{HealthCheckRequest, HealthCheckResponse};

use crate::bridge::PyBridge;
use crate::proto::sglang_service_server::SglangServiceServer;
use crate::server::SglangServiceImpl;

const SGLANG_SERVICE: &str = <SglangServiceServer<SglangServiceImpl> as NamedService>::NAME;

pub(crate) struct StandardHealthService {
    check: Arc<dyn Fn() -> bool + Send + Sync>,
    stopping: watch::Receiver<bool>,
}

impl StandardHealthService {
    pub(crate) fn new(bridge: Arc<PyBridge>, stopping: watch::Receiver<bool>) -> Self {
        Self {
            check: Arc::new(move || match bridge.health_check() {
                Ok(healthy) => healthy,
                Err(err) => {
                    tracing::warn!("gRPC health check failed: {err}");
                    false
                }
            }),
            stopping,
        }
    }

    async fn serving_status(&self) -> ServingStatus {
        let mut stopping = self.stopping.clone();
        tokio::select! {
            biased;
            _ = stopping.wait_for(|stopping| *stopping) => ServingStatus::NotServing,
            result = async {
                let check = self.check.clone();
                tokio::task::spawn_blocking(move || check()).await
            } => match result {
                Ok(true) => ServingStatus::Serving,
                Ok(false) => ServingStatus::NotServing,
                Err(err) => {
                    tracing::warn!("gRPC health check task failed: {err}");
                    ServingStatus::NotServing
                }
            }
        }
    }
}

#[tonic::async_trait]
impl Health for StandardHealthService {
    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        if !matches!(request.get_ref().service.as_str(), "" | SGLANG_SERVICE) {
            return Err(Status::not_found("service not registered"));
        }
        Ok(Response::new(HealthCheckResponse {
            status: self.serving_status().await as i32,
        }))
    }

    type WatchStream = tokio_stream::Empty<Result<HealthCheckResponse, Status>>;

    async fn watch(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        Err(Status::unimplemented("Watch is not supported; use Check"))
    }
}

#[cfg(test)]
mod tests;
