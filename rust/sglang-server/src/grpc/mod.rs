//! Tonic adapter for the native `sglang.api.v1` service.
//!
//! The service is the gRPC rendering of the contract the native HTTP endpoints
//! serve: `Generate` is `/generate`, `HealthCheck` is `/health_generate`, and
//! `GetModelInfo` / `GetServerInfo` are their HTTP namesakes. Requests decode
//! through the shared `/generate` fan-out and responses stream through the
//! shared [`FrontendHandle`], which owns preprocessing, admission, runtime
//! communication, and request cancellation.
//!
//! The thin Tonic-service structure, streamed response approach, and test
//! strategy build on Rain Jiang's multi-protocol prototype in
//! `sgl-project/sglang#36923`.

use std::pin::Pin;
use std::time::{Duration, Instant};

use futures::Stream;
use sglang_api_types::api::v1 as api;
use sglang_api_types::api::v1::sglang_service_server::SglangService;
use tonic::{Request, Response, Status};

use crate::frontend::{FrontendHandle, HealthStatus};
use crate::message::config::{PreferredSamplingParams, ServerArgs};
use crate::message::request::into_requests;
use crate::message::wire::fill_preferred_sampling;
use crate::utils::environ;

mod info;
mod response;
mod server;

pub(crate) use server::serve;

#[cfg(test)]
mod tests;

const DEFAULT_RESPONSE_TIMEOUT: Duration = Duration::from_secs(300);

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// Narrow snapshot of launch policy the adapter needs per call. It
/// intentionally excludes listener, authentication, TLS, and lifecycle
/// configuration.
#[derive(Clone)]
struct AdapterConfig {
    preferred_sampling_params: Option<PreferredSamplingParams>,
    incremental_streaming_output: bool,
    /// Longest wait for the next event of any in-flight call before the RPC
    /// ends with DEADLINE_EXCEEDED.
    response_timeout: Duration,
    /// `/health_generate`'s heartbeat window (`SGLANG_HEALTH_CHECK_TIMEOUT`).
    health_timeout: Duration,
}

/// Tonic-facing implementation backed by the transport-neutral Rust frontend.
pub(crate) struct GrpcService {
    frontend: FrontendHandle,
    config: AdapterConfig,
}

impl GrpcService {
    pub(crate) fn new(frontend: FrontendHandle, server_args: &ServerArgs) -> Self {
        Self {
            frontend,
            config: AdapterConfig {
                preferred_sampling_params: server_args.preferred_sampling_params.clone(),
                incremental_streaming_output: server_args.incremental_streaming_output,
                response_timeout: DEFAULT_RESPONSE_TIMEOUT,
                health_timeout: Duration::from_secs(
                    environ::env_i64("SGLANG_HEALTH_CHECK_TIMEOUT", 20).max(0) as u64,
                ),
            },
        }
    }

    #[cfg(test)]
    fn for_test(
        frontend: FrontendHandle,
        preferred_sampling_params: Option<PreferredSamplingParams>,
        incremental_streaming_output: bool,
        response_timeout: Duration,
    ) -> Self {
        Self {
            frontend,
            config: AdapterConfig {
                preferred_sampling_params,
                incremental_streaming_output,
                response_timeout,
                health_timeout: Duration::from_millis(20),
            },
        }
    }
}

#[tonic::async_trait]
impl SglangService for GrpcService {
    type GenerateStream = ResponseStream<api::GenerateStreamItem>;

    /// `/generate` over gRPC: one schema, one fan-out, one admission path. A
    /// body that fails normalization ends the RPC with INVALID_ARGUMENT before
    /// anything reaches the scheduler, as the HTTP 400 does.
    async fn generate(
        &self,
        request: Request<api::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let mut request = request.into_inner();
        if let Some(preferred) = &self.config.preferred_sampling_params {
            request.sampling_params =
                fill_preferred_sampling(request.sampling_params.take(), &preferred.0)
                    .map_err(Status::internal)?;
        }
        let stream = request.stream.unwrap_or(false);
        let (payloads, is_batch) =
            into_requests(request).map_err(|error| Status::invalid_argument(error.to_string()))?;
        // Python starts its request clock after normalization and before
        // tokenization / multimodal preprocessing; `e2e_latency` measures from
        // the same boundary here.
        let created_at = Instant::now();
        let calls = self
            .frontend
            .generate_batch(payloads)
            .await
            .map_err(response::status)?;
        Ok(Response::new(response::generate_stream(
            calls,
            response::StreamOptions {
                stream,
                incremental: self.config.incremental_streaming_output,
                with_index: is_batch,
                response_timeout: self.config.response_timeout,
                created_at,
            },
        )))
    }

    async fn health_check(
        &self,
        _request: Request<api::HealthCheckRequest>,
    ) -> Result<Response<api::HealthCheckResponse>, Status> {
        match self.frontend.probe_health(self.config.health_timeout).await {
            Ok(HealthStatus::Healthy) => {
                Ok(Response::new(api::HealthCheckResponse { healthy: true }))
            }
            Ok(HealthStatus::NotReady) => Err(Status::unavailable(
                "server is still completing its startup warmup",
            )),
            Ok(HealthStatus::Stalled) => Err(Status::unavailable(
                "no scheduler output within the health-check timeout",
            )),
            Err(error) => Err(response::status(error)),
        }
    }

    async fn get_model_info(
        &self,
        _request: Request<api::GetModelInfoRequest>,
    ) -> Result<Response<api::GetModelInfoResponse>, Status> {
        info::model_info(self.frontend.model_info())
            .map(Response::new)
            .map_err(Status::internal)
    }

    async fn get_server_info(
        &self,
        _request: Request<api::GetServerInfoRequest>,
    ) -> Result<Response<api::GetServerInfoResponse>, Status> {
        let server_info = self
            .frontend
            .server_info()
            .await
            .map_err(response::status)?;
        info::server_info(server_info)
            .map(Response::new)
            .map_err(Status::internal)
    }
}
