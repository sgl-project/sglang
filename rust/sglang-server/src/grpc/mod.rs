//! Tonic adapter for the canonical `sglang.runtime.v1` API.
//!
//! This module translates protobuf requests and responses and mounts that thin
//! adapter on Tonic. The shared [`FrontendHandle`] owns preprocessing,
//! admission, runtime communication, and request cancellation.
//!
//! The thin Tonic-service structure, streamed response approach, and test
//! strategy build on Rain Jiang's multi-protocol prototype in
//! `sgl-project/sglang#36923`. This stack keeps those ideas while using the
//! canonical `runtime.v1` contract and the narrower [`FrontendHandle`] seam.

use std::pin::Pin;
use std::time::Duration;

use futures::Stream;
use sglang_grpc_types::sglang::runtime::v1 as proto;
use sglang_grpc_types::sglang::runtime::v1::sglang_service_server::SglangService;
use tonic::{Request, Response, Status};

use crate::frontend::FrontendHandle;
use crate::message::config::{PreferredSamplingParams, ServerArgs};

mod convert;
mod response;
mod server;

pub(crate) use server::serve;

#[cfg(test)]
mod tests;

const DEFAULT_RESPONSE_TIMEOUT: Duration = Duration::from_secs(300);

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// Narrow snapshot of launch policy required to preserve the existing
/// `runtime.v1` generation behavior. It intentionally excludes listener,
/// authentication, TLS, and lifecycle configuration.
#[derive(Clone)]
struct AdapterConfig {
    preferred_sampling_params: Option<PreferredSamplingParams>,
    incremental_streaming_output: bool,
    response_timeout: Duration,
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
            },
        }
    }
}

fn unimplemented_rpc(name: &'static str) -> Status {
    Status::unimplemented(format!(
        "{name} is not implemented by the Rust frontend gRPC adapter"
    ))
}

#[tonic::async_trait]
impl SglangService for GrpcService {
    async fn text_generate(
        &self,
        request: Request<proto::TextGenerateRequest>,
    ) -> Result<Response<ResponseStream<proto::TextGenerateResponse>>, Status> {
        let request = convert::text_generate(
            request.into_inner(),
            self.config.preferred_sampling_params.as_ref(),
        )
        .map_err(Status::from)?;
        let call = self
            .frontend
            .generate(request)
            .await
            .map_err(response::status)?;
        Ok(Response::new(response::text_generate_stream(
            call,
            self.config.incremental_streaming_output,
            self.config.response_timeout,
        )))
    }

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<ResponseStream<proto::GenerateResponse>>, Status> {
        let request = convert::generate(
            request.into_inner(),
            self.config.preferred_sampling_params.as_ref(),
        )
        .map_err(Status::from)?;
        let call = self
            .frontend
            .generate(request)
            .await
            .map_err(response::status)?;
        Ok(Response::new(response::generate_stream(
            call,
            self.config.incremental_streaming_output,
            self.config.response_timeout,
        )))
    }

    // runtime.v1 is shared with the broader Python-backed service, so its
    // generated trait contains more operations than this initial Rust adapter.
    // Keep that support boundary explicit: adding an RPC is a deliberate future
    // change, never an accidental empty success.
    async fn text_embed(
        &self,
        _request: Request<proto::TextEmbedRequest>,
    ) -> Result<Response<proto::TextEmbedResponse>, Status> {
        Err(unimplemented_rpc("text_embed"))
    }

    async fn embed(
        &self,
        _request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        Err(unimplemented_rpc("embed"))
    }

    async fn classify(
        &self,
        _request: Request<proto::ClassifyRequest>,
    ) -> Result<Response<proto::ClassifyResponse>, Status> {
        Err(unimplemented_rpc("classify"))
    }

    async fn tokenize(
        &self,
        _request: Request<proto::TokenizeRequest>,
    ) -> Result<Response<proto::TokenizeResponse>, Status> {
        Err(unimplemented_rpc("tokenize"))
    }

    async fn detokenize(
        &self,
        _request: Request<proto::DetokenizeRequest>,
    ) -> Result<Response<proto::DetokenizeResponse>, Status> {
        Err(unimplemented_rpc("detokenize"))
    }

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        Err(unimplemented_rpc("health_check"))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        Err(unimplemented_rpc("get_model_info"))
    }

    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        Err(unimplemented_rpc("get_server_info"))
    }

    async fn list_models(
        &self,
        _request: Request<proto::ListModelsRequest>,
    ) -> Result<Response<proto::ListModelsResponse>, Status> {
        Err(unimplemented_rpc("list_models"))
    }

    async fn get_load(
        &self,
        _request: Request<proto::GetLoadRequest>,
    ) -> Result<Response<proto::GetLoadResponse>, Status> {
        Err(unimplemented_rpc("get_load"))
    }

    async fn abort(
        &self,
        _request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        Err(unimplemented_rpc("abort"))
    }

    async fn flush_cache(
        &self,
        _request: Request<proto::FlushCacheRequest>,
    ) -> Result<Response<proto::FlushCacheResponse>, Status> {
        Err(unimplemented_rpc("flush_cache"))
    }

    async fn pause_generation(
        &self,
        _request: Request<proto::PauseGenerationRequest>,
    ) -> Result<Response<proto::PauseGenerationResponse>, Status> {
        Err(unimplemented_rpc("pause_generation"))
    }

    async fn watch_engine_state(
        &self,
        _request: Request<proto::WatchEngineStateRequest>,
    ) -> Result<Response<ResponseStream<proto::EngineStateSnapshot>>, Status> {
        Err(unimplemented_rpc("watch_engine_state"))
    }

    async fn continue_generation(
        &self,
        _request: Request<proto::ContinueGenerationRequest>,
    ) -> Result<Response<proto::ContinueGenerationResponse>, Status> {
        Err(unimplemented_rpc("continue_generation"))
    }

    async fn chat_complete(
        &self,
        _request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<ResponseStream<proto::OpenAiStreamChunk>>, Status> {
        Err(unimplemented_rpc("chat_complete"))
    }

    async fn complete(
        &self,
        _request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<ResponseStream<proto::OpenAiStreamChunk>>, Status> {
        Err(unimplemented_rpc("complete"))
    }

    async fn open_ai_embed(
        &self,
        _request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        Err(unimplemented_rpc("open_ai_embed"))
    }

    async fn open_ai_classify(
        &self,
        _request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        Err(unimplemented_rpc("open_ai_classify"))
    }

    async fn score(
        &self,
        _request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        Err(unimplemented_rpc("score"))
    }

    async fn rerank(
        &self,
        _request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        Err(unimplemented_rpc("rerank"))
    }

    async fn start_profile(
        &self,
        _request: Request<proto::StartProfileRequest>,
    ) -> Result<Response<proto::StartProfileResponse>, Status> {
        Err(unimplemented_rpc("start_profile"))
    }

    async fn stop_profile(
        &self,
        _request: Request<proto::StopProfileRequest>,
    ) -> Result<Response<proto::StopProfileResponse>, Status> {
        Err(unimplemented_rpc("stop_profile"))
    }

    async fn update_weights_from_disk(
        &self,
        _request: Request<proto::UpdateWeightsRequest>,
    ) -> Result<Response<proto::UpdateWeightsResponse>, Status> {
        Err(unimplemented_rpc("update_weights_from_disk"))
    }
}
