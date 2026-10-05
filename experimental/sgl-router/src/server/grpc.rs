// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The engine's native gRPC interface (`sglang.runtime.v1.SglangService`),
//! served so a gRPC client can point at the router or at an engine. Each served
//! RPC is routed like its HTTP route and forwarded to the same RPC on an engine.

use crate::proxy::grpc::ChunkStream;
use crate::server::app::pod_id;
use crate::server::app_context::AppContext;
use crate::server::metrics::{outcome_from_status, RequestLogContext};
use crate::server::routes::chat::{chat_completions_grpc, MAX_CHAT_BODY_BYTES};
use crate::workers::GRPC_MAX_MESSAGE_BYTES;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use bytes::Bytes;
use futures::StreamExt;
use sglang_grpc_types::sglang::runtime::v1 as proto;
use sglang_grpc_types::sglang::runtime::v1::sglang_service_server::{
    SglangService, SglangServiceServer,
};
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tonic::{Request, Response, Status};

const CHAT_COMPLETE_ROUTE: &str = "/sglang.runtime.v1.SglangService/ChatComplete";

type RpcResult<T> = Result<Response<T>, Status>;
type Unserved<T> = futures::stream::Empty<Result<T, Status>>;

fn unserved<T>(rpc: &str) -> RpcResult<T> {
    Err(Status::unimplemented(format!(
        "{rpc} is not served by sgl-router"
    )))
}

/// Serve until `shutdown` resolves, then let open calls finish.
pub async fn serve(
    listener: TcpListener,
    ctx: Arc<AppContext>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let service = SglangServiceServer::new(RouterGrpc { ctx })
        .max_decoding_message_size(MAX_CHAT_BODY_BYTES)
        .max_encoding_message_size(GRPC_MAX_MESSAGE_BYTES);
    tonic::transport::Server::builder()
        .add_service(service)
        .serve_with_incoming_shutdown(
            tokio_stream::wrappers::TcpListenerStream::new(listener),
            shutdown,
        )
        .await?;
    Ok(())
}

struct RouterGrpc {
    ctx: Arc<AppContext>,
}

/// Client metadata plus `trace_headers`, which the engine reads as request
/// headers, so header-driven routing (sticky keys, sessions, SLOs) works as over HTTP.
fn request_headers(
    metadata: tonic::metadata::MetadataMap,
    request: &proto::OpenAiRequest,
) -> HeaderMap {
    let mut headers = metadata.into_headers();
    for (name, value) in &request.trace_headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::try_from(name.as_str()),
            HeaderValue::try_from(value.as_str()),
        ) {
            headers.insert(name, value);
        }
    }
    headers
}

/// The access-log line the HTTP edge middleware writes, for a gRPC call.
fn log_call(
    route: &str,
    headers: &HeaderMap,
    status: u16,
    log_context: Option<&RequestLogContext>,
    started: Instant,
) {
    tracing::info!(
        pod_id = %pod_id(),
        request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("-"),
        path = route,
        status,
        outcome = log_context
            .map(|c| c.outcome)
            .unwrap_or_else(|| outcome_from_status(status))
            .as_str(),
        worker = log_context.map(|c| c.worker_url.as_str()).unwrap_or(""),
        engine_rid = log_context
            .and_then(|c| c.engine_rid.as_deref())
            .unwrap_or(""),
        model = log_context.map(|c| c.model_id.as_str()).unwrap_or(""),
        stream = log_context.is_some_and(|c| c.streaming),
        latency_ms = started.elapsed().as_millis() as u64,
        "grpc_request",
    );
}

impl RouterGrpc {
    async fn chat_complete(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> RpcResult<ChunkStream> {
        let started = Instant::now();
        let ctx = &self.ctx;
        ctx.metrics.record_ingress(CHAT_COMPLETE_ROUTE, "POST");
        let inflight = ctx.inflight_http.enter();
        let (metadata, _, request) = request.into_parts();
        let headers = request_headers(metadata, &request);
        let (result, log_context) =
            chat_completions_grpc(ctx, headers.clone(), Bytes::from(request.json_body)).await;
        let status = match &result {
            Ok(response) => response.status.as_u16(),
            Err(error) => error.status_code().as_u16(),
        };
        ctx.metrics
            .record_response(CHAT_COMPLETE_ROUTE, "POST", status);
        log_call(
            CHAT_COMPLETE_ROUTE,
            &headers,
            status,
            log_context.as_ref(),
            started,
        );
        let chunks = result.map_err(|error| error.into_grpc_status())?.chunks;
        // Counted until the client's stream ends, as an HTTP body is.
        let chunks = chunks.map(move |chunk| {
            let _ = &inflight;
            chunk
        });
        Ok(Response::new(Box::pin(chunks)))
    }
}

/// Implements `SglangService` from the served items; every listed RPC answers
/// `Unimplemented`. A macro because `async_trait` must see the expanded methods.
macro_rules! sglang_service {
    (
        served { $($served:tt)* }
        unserved { $($rpc:ident($req:ident) -> $resp:ident;)* }
        unserved_streams { $($srpc:ident($sreq:ident) -> $stream:ident<$sresp:ident>;)* }
    ) => {
        #[tonic::async_trait]
        impl SglangService for RouterGrpc {
            $($served)*
            $(
                async fn $rpc(&self, _: Request<proto::$req>) -> RpcResult<proto::$resp> {
                    unserved(stringify!($rpc))
                }
            )*
            $(
                type $stream = Unserved<proto::$sresp>;
                async fn $srpc(&self, _: Request<proto::$sreq>) -> RpcResult<Self::$stream> {
                    unserved(stringify!($srpc))
                }
            )*
        }
    };
}

sglang_service! {
    served {
        type ChatCompleteStream = ChunkStream;
        async fn chat_complete(&self, request: Request<proto::OpenAiRequest>) -> RpcResult<ChunkStream> {
            RouterGrpc::chat_complete(self, request).await
        }
    }
    unserved {
        text_embed(TextEmbedRequest) -> TextEmbedResponse;
        embed(EmbedRequest) -> EmbedResponse;
        classify(ClassifyRequest) -> ClassifyResponse;
        tokenize(TokenizeRequest) -> TokenizeResponse;
        detokenize(DetokenizeRequest) -> DetokenizeResponse;
        health_check(HealthCheckRequest) -> HealthCheckResponse;
        get_model_info(GetModelInfoRequest) -> GetModelInfoResponse;
        get_server_info(GetServerInfoRequest) -> GetServerInfoResponse;
        list_models(ListModelsRequest) -> ListModelsResponse;
        get_load(GetLoadRequest) -> GetLoadResponse;
        abort(AbortRequest) -> AbortResponse;
        flush_cache(FlushCacheRequest) -> FlushCacheResponse;
        pause_generation(PauseGenerationRequest) -> PauseGenerationResponse;
        continue_generation(ContinueGenerationRequest) -> ContinueGenerationResponse;
        open_ai_embed(OpenAiRequest) -> OpenAiResponse;
        open_ai_classify(OpenAiRequest) -> OpenAiResponse;
        score(OpenAiRequest) -> OpenAiResponse;
        rerank(OpenAiRequest) -> OpenAiResponse;
        start_profile(StartProfileRequest) -> StartProfileResponse;
        stop_profile(StopProfileRequest) -> StopProfileResponse;
        update_weights_from_disk(UpdateWeightsRequest) -> UpdateWeightsResponse;
    }
    unserved_streams {
        text_generate(TextGenerateRequest) -> TextGenerateStream<TextGenerateResponse>;
        generate(GenerateRequest) -> GenerateStream<GenerateResponse>;
        complete(OpenAiRequest) -> CompleteStream<OpenAiStreamChunk>;
        watch_engine_state(WatchEngineStateRequest) -> WatchEngineStateStream<EngineStateSnapshot>;
    }
}
