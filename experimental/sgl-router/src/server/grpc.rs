// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The engine's native gRPC interface (`sglang.runtime.v1.SglangService`),
//! served so a gRPC client can point at the router or at an engine. Each served
//! RPC is routed like its HTTP route and forwarded to the same RPC on an engine.

use crate::proxy::grpc::{
    embed_to_text, generate_to_text, text_to_embed, text_to_generate, GrpcResponse, OpenAiRpc,
    OpenAiUnaryRpc, Replies,
};
use crate::server::app::pod_id;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{outcome_from_status, RequestLogContext};
use crate::server::routes::chat::{
    prompt_ids, route_grpc, route_typed, Endpoint, MAX_CHAT_BODY_BYTES,
};
use crate::workers::GRPC_MAX_MESSAGE_BYTES;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use bytes::Bytes;
use futures::StreamExt;
use sglang_grpc_types::sglang::runtime::v1 as proto;
use sglang_grpc_types::sglang::runtime::v1::sglang_service_server::{
    SglangService, SglangServiceServer,
};
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tonic::{Request, Response, Status};

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

type Routed<T> = (Result<GrpcResponse<T>, ApiError>, Option<RequestLogContext>);

/// Client metadata plus `trace_headers`, which the engine reads as request
/// headers, so header-driven routing (sticky keys, sessions, SLOs) works as over HTTP.
fn request_headers<T>(
    request: Request<T>,
    trace_headers: impl Fn(&T) -> &HashMap<String, String>,
) -> (HeaderMap, T) {
    let (metadata, _, request) = request.into_parts();
    let mut headers = metadata.into_headers();
    for (name, value) in trace_headers(&request) {
        if let (Ok(name), Ok(value)) = (
            HeaderName::try_from(name.as_str()),
            HeaderValue::try_from(value.as_str()),
        ) {
            headers.insert(name, value);
        }
    }
    (headers, request)
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
    /// Run a routed call with the HTTP edge's metrics, in-flight count and access log.
    async fn finish<T: Send + 'static>(
        &self,
        rpc: &str,
        headers: HeaderMap,
        call: impl Future<Output = Routed<T>>,
    ) -> Result<Replies<T>, Status> {
        let started = Instant::now();
        let route = format!("/sglang.runtime.v1.SglangService/{rpc}");
        let metrics = &self.ctx.metrics;
        metrics.record_ingress(&route, "POST");
        let inflight = self.ctx.inflight_http.enter();
        let (result, log_context) = call.await;
        let status = match &result {
            Ok(response) => response.status.as_u16(),
            Err(error) => error.status_code().as_u16(),
        };
        metrics.record_response(&route, "POST", status);
        log_call(&route, &headers, status, log_context.as_ref(), started);
        let replies = result.map_err(ApiError::into_grpc_status)?.replies;
        // Counted until the client's stream ends, as an HTTP body is.
        Ok(replies
            .map(move |reply| {
                let _ = &inflight;
                reply
            })
            .boxed())
    }

    /// [`Self::finish`] for a unary RPC, whose one reply the engine sent.
    async fn finish_unary<T: Send + 'static>(
        &self,
        rpc: &str,
        headers: HeaderMap,
        call: impl Future<Output = Routed<T>>,
    ) -> RpcResult<T> {
        let mut replies = self.finish(rpc, headers, call).await?;
        let reply = replies.next().await;
        reply
            .unwrap_or_else(|| Err(Status::internal("the engine sent no reply")))
            .map(Response::new)
    }

    async fn openai(
        &self,
        rpc: OpenAiRpc,
        endpoint: Endpoint,
        request: Request<proto::OpenAiRequest>,
    ) -> RpcResult<Replies<proto::OpenAiStreamChunk>> {
        let (headers, request) = request_headers(request, |r| &r.trace_headers);
        let body = Bytes::from(request.json_body);
        let call = route_grpc(&self.ctx, endpoint, rpc, headers.clone(), body);
        let name = match rpc {
            OpenAiRpc::ChatComplete => "ChatComplete",
            OpenAiRpc::Complete => "Complete",
        };
        self.finish(name, headers, call).await.map(Response::new)
    }

    /// Tokenized here, as `/generate` tokenizes `text`, then sent as `Generate` for its text.
    async fn text_generate(
        &self,
        request: Request<proto::TextGenerateRequest>,
    ) -> Result<Replies<proto::TextGenerateResponse>, Status> {
        let (headers, request) = request_headers(request, |r| &r.trace_headers);
        let ids = (request.return_text_in_logprobs != Some(true))
            .then(|| prompt_ids(&self.ctx, Endpoint::Generate, &request.text))
            .flatten();
        let Some(ids) = ids else {
            let call = route_typed(&self.ctx, Endpoint::Generate, request, headers.clone());
            return self.finish("TextGenerate", headers, call).await;
        };
        let request = text_to_generate(request, ids);
        let call = route_typed(&self.ctx, Endpoint::Generate, request, headers.clone());
        let replies = self.finish("TextGenerate", headers, call).await?;
        Ok(replies.map(|reply| reply.map(generate_to_text)).boxed())
    }

    /// Tokenized here, as `/v1/embeddings` tokenizes text, then sent as `Embed`.
    async fn text_embed(
        &self,
        request: Request<proto::TextEmbedRequest>,
    ) -> RpcResult<proto::TextEmbedResponse> {
        let (headers, request) = request_headers(request, |r| &r.trace_headers);
        let Some(ids) = prompt_ids(&self.ctx, Endpoint::Embeddings, &request.text) else {
            let call = route_typed(&self.ctx, Endpoint::Embeddings, request, headers.clone());
            return self.finish_unary("TextEmbed", headers, call).await;
        };
        let call = route_typed(
            &self.ctx,
            Endpoint::Embeddings,
            text_to_embed(request, ids),
            headers.clone(),
        );
        let reply = self.finish_unary("TextEmbed", headers, call).await?;
        Ok(reply.map(embed_to_text))
    }

    async fn openai_unary(
        &self,
        rpc: OpenAiUnaryRpc,
        endpoint: Endpoint,
        request: Request<proto::OpenAiRequest>,
    ) -> RpcResult<proto::OpenAiResponse> {
        let (headers, request) = request_headers(request, |r| &r.trace_headers);
        let body = Bytes::from(request.json_body);
        let call = route_grpc(&self.ctx, endpoint, rpc, headers.clone(), body);
        let name = match rpc {
            OpenAiUnaryRpc::Embed => "OpenAIEmbed",
            OpenAiUnaryRpc::Classify => "OpenAIClassify",
            OpenAiUnaryRpc::Rerank => "Rerank",
        };
        self.finish_unary(name, headers, call).await
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

/// A typed RPC: prepared through the JSON its HTTP sibling reads, then sent as typed.
macro_rules! typed {
    ($self:ident, $request:ident, $endpoint:ident, $rpc:literal, $finish:ident) => {{
        let (headers, request) = request_headers($request, |r| &r.trace_headers);
        let call = route_typed(&$self.ctx, Endpoint::$endpoint, request, headers.clone());
        $self.$finish($rpc, headers, call).await
    }};
}

sglang_service! {
    served {
        type ChatCompleteStream = Replies<proto::OpenAiStreamChunk>;
        async fn chat_complete(&self, request: Request<proto::OpenAiRequest>) -> RpcResult<Self::ChatCompleteStream> {
            self.openai(OpenAiRpc::ChatComplete, Endpoint::Chat, request).await
        }

        type CompleteStream = Replies<proto::OpenAiStreamChunk>;
        async fn complete(&self, request: Request<proto::OpenAiRequest>) -> RpcResult<Self::CompleteStream> {
            self.openai(OpenAiRpc::Complete, Endpoint::Completions, request).await
        }

        type GenerateStream = Replies<proto::GenerateResponse>;
        async fn generate(&self, request: Request<proto::GenerateRequest>) -> RpcResult<Self::GenerateStream> {
            typed!(self, request, Generate, "Generate", finish).map(Response::new)
        }

        type TextGenerateStream = Replies<proto::TextGenerateResponse>;
        async fn text_generate(&self, request: Request<proto::TextGenerateRequest>) -> RpcResult<Self::TextGenerateStream> {
            self.text_generate(request).await.map(Response::new)
        }

        async fn embed(&self, request: Request<proto::EmbedRequest>) -> RpcResult<proto::EmbedResponse> {
            typed!(self, request, Embeddings, "Embed", finish_unary)
        }

        async fn text_embed(&self, request: Request<proto::TextEmbedRequest>) -> RpcResult<proto::TextEmbedResponse> {
            self.text_embed(request).await
        }

        async fn classify(&self, mut request: Request<proto::ClassifyRequest>) -> RpcResult<proto::ClassifyResponse> {
            // A text prompt is sent as ids, as `/v1/classify` forwards it.
            let classify = request.get_mut();
            if classify.input_ids.is_empty() {
                if let Some(ids) = prompt_ids(&self.ctx, Endpoint::Classify, &classify.text) {
                    (classify.input_ids, classify.text) = (ids, String::new());
                }
            }
            typed!(self, request, Classify, "Classify", finish_unary)
        }

        async fn open_ai_embed(&self, request: Request<proto::OpenAiRequest>) -> RpcResult<proto::OpenAiResponse> {
            self.openai_unary(OpenAiUnaryRpc::Embed, Endpoint::Embeddings, request).await
        }

        async fn open_ai_classify(&self, request: Request<proto::OpenAiRequest>) -> RpcResult<proto::OpenAiResponse> {
            self.openai_unary(OpenAiUnaryRpc::Classify, Endpoint::Classify, request).await
        }

        async fn rerank(&self, request: Request<proto::OpenAiRequest>) -> RpcResult<proto::OpenAiResponse> {
            self.openai_unary(OpenAiUnaryRpc::Rerank, Endpoint::Rerank, request).await
        }
    }
    unserved {
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
        score(OpenAiRequest) -> OpenAiResponse;
        start_profile(StartProfileRequest) -> StartProfileResponse;
        stop_profile(StopProfileRequest) -> StopProfileResponse;
        update_weights_from_disk(UpdateWeightsRequest) -> UpdateWeightsResponse;
    }
    unserved_streams {
        watch_engine_state(WatchEngineStateRequest) -> WatchEngineStateStream<EngineStateSnapshot>;
    }
}
