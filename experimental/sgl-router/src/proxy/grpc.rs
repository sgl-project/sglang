// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Forwarding to an engine's native gRPC `SglangService`, for gRPC clients.

use super::sse::{
    self, ErrorEventScanner, PumpItem, RouterStreamError, StreamEnd, StreamEndReason, StreamLimits,
};
use super::{breaker_outcome, stream_breaker_outcome, BreakerOutcome, Proxy};
use crate::config::SamplingField;
use crate::server::error::{router_grpc_status, ApiError};
use crate::server::header_utils::should_forward_request_header;
use crate::workers::Worker;
use axum::http::{HeaderMap, StatusCode};
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, StreamExt, TryStreamExt};
use serde_json::Value;
use sglang_grpc_types::sglang::runtime::v1 as proto;
use sglang_grpc_types::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;
use tonic::{Code, Status};

pub type Replies<T> = BoxStream<'static, Result<T, Status>>;

/// An engine's answer to a gRPC call; a unary reply is a one-item stream.
pub struct GrpcResponse<T> {
    /// The engine's HTTP status for an OpenAI RPC that reports one; otherwise 200.
    pub status: StatusCode,
    pub replies: Replies<T>,
}

/// One engine RPC, sent with the body the router prepared for its HTTP sibling.
pub trait EngineRpc: Send + Sync + 'static {
    type Reply: PumpItem;

    /// Call the engine: OpenAI RPCs send the prepared `body`, typed ones apply `additions`.
    fn call(
        &self,
        client: SglangServiceClient<Channel>,
        body: Bytes,
        additions: &Additions,
        trace_headers: HashMap<String, String>,
    ) -> RpcFuture<Self::Reply>;

    /// The HTTP status a reply carries, for RPCs that report one.
    fn status(_reply: &Self::Reply) -> Option<StatusCode> {
        None
    }
}

impl PumpItem for proto::OpenAiStreamChunk {
    fn is_error_event(&self, _: &mut ErrorEventScanner) -> bool {
        sse::is_error_payload(&self.json_chunk)
    }
}

impl PumpItem for proto::GenerateResponse {
    fn is_error_event(&self, _: &mut ErrorEventScanner) -> bool {
        is_error_abort(&self.meta_info)
    }
}

impl PumpItem for proto::TextGenerateResponse {
    fn is_error_event(&self, _: &mut ErrorEventScanner) -> bool {
        is_error_abort(&self.meta_info)
    }
}

impl PumpItem for proto::OpenAiResponse {}
impl PumpItem for proto::EmbedResponse {}
impl PumpItem for proto::TextEmbedResponse {}
impl PumpItem for proto::ClassifyResponse {}

/// An engine abort with a status code, as native `/generate` reports a failure;
/// a user abort has none. `meta_info` values are JSON-encoded.
fn is_error_abort(meta_info: &HashMap<String, String>) -> bool {
    let reason = meta_info.get("finish_reason");
    let reason = reason.and_then(|reason| serde_json::from_str::<Value>(reason).ok());
    reason.is_some_and(|reason| reason["type"] == "abort" && reason["status_code"].is_u64())
}

fn http_status(code: i32) -> Option<StatusCode> {
    StatusCode::from_u16(u16::try_from(code).ok()?).ok()
}

/// The streaming OpenAI RPCs, whose request is the JSON body the HTTP route forwards.
#[derive(Clone, Copy)]
pub enum OpenAiRpc {
    ChatComplete,
    Complete,
}

impl EngineRpc for OpenAiRpc {
    type Reply = proto::OpenAiStreamChunk;

    fn call(
        &self,
        mut client: SglangServiceClient<Channel>,
        body: Bytes,
        _: &Additions,
        trace_headers: HashMap<String, String>,
    ) -> RpcFuture<Self::Reply> {
        let request = proto::OpenAiRequest {
            json_body: body.into(),
            trace_headers,
        };
        let rpc = *self;
        Box::pin(async move {
            let response = match rpc {
                OpenAiRpc::ChatComplete => client.chat_complete(request).await?,
                OpenAiRpc::Complete => client.complete(request).await?,
            };
            Ok(response.into_inner().boxed())
        })
    }

    fn status(reply: &Self::Reply) -> Option<StatusCode> {
        reply.status_code.and_then(http_status)
    }
}

/// The unary OpenAI RPCs, whose request is the JSON body the HTTP route forwards.
#[derive(Clone, Copy)]
pub enum OpenAiUnaryRpc {
    Embed,
    Classify,
    Rerank,
}

impl EngineRpc for OpenAiUnaryRpc {
    type Reply = proto::OpenAiResponse;

    fn call(
        &self,
        mut client: SglangServiceClient<Channel>,
        body: Bytes,
        _: &Additions,
        trace_headers: HashMap<String, String>,
    ) -> RpcFuture<Self::Reply> {
        let request = proto::OpenAiRequest {
            json_body: body.into(),
            trace_headers,
        };
        let rpc = *self;
        Box::pin(async move {
            let response = match rpc {
                OpenAiUnaryRpc::Embed => client.open_ai_embed(request).await?,
                OpenAiUnaryRpc::Classify => client.open_ai_classify(request).await?,
                OpenAiUnaryRpc::Rerank => client.rerank(request).await?,
            };
            Ok(futures::stream::iter([Ok(response.into_inner())]).boxed())
        })
    }

    fn status(reply: &Self::Reply) -> Option<StatusCode> {
        http_status(reply.status_code)
    }
}

/// What routing reads from a typed request, in place of its HTTP route's body.
pub struct RoutingView<'a> {
    pub input_ids: &'a [i32],
    /// The prompt, when it was sent as text rather than `input_ids`.
    pub text: Option<&'a str>,
    pub sampling_params: Option<&'a proto::SamplingParams>,
    pub stream: bool,
    pub rid: Option<&'a str>,
}

/// What the router adds to a typed request, as it adds it to an HTTP body.
#[derive(Clone, Default)]
pub struct Additions {
    pub rid: Option<String>,
    pub bootstrap: Option<proto::DisaggregatedParams>,
    pub sampling_defaults: Vec<(SamplingField, f64)>,
    /// Set when the router owns the DP rank; `Some(None)` unpins it.
    pub routed_dp_rank: Option<Option<u32>>,
}

/// A typed RPC, prepared from its fields like its HTTP route.
pub trait TypedRpc: Clone + Send + Sync + 'static {
    type Reply: PumpItem;

    fn view(&self) -> RoutingView<'_>;

    /// Take the router's additions; a caller's own values win.
    fn apply(&mut self, additions: &Additions);

    /// The trace context the engine propagates.
    fn trace_headers(&mut self) -> &mut HashMap<String, String>;

    fn send(self, client: SglangServiceClient<Channel>) -> RpcFuture<Self::Reply>;
}

type RpcFuture<T> = BoxFuture<'static, Result<Replies<T>, Status>>;

impl<T: TypedRpc> EngineRpc for T {
    type Reply = T::Reply;

    fn call(
        &self,
        client: SglangServiceClient<Channel>,
        _: Bytes,
        additions: &Additions,
        trace_headers: HashMap<String, String>,
    ) -> RpcFuture<T::Reply> {
        let mut request = self.clone();
        request.apply(additions);
        // Headers sent as gRPC metadata travel on as trace context; a credential does not.
        let forwarded = trace_headers
            .into_iter()
            .filter(|(name, _)| name != "authorization");
        for (name, value) in forwarded {
            request.trace_headers().entry(name).or_insert(value);
        }
        request.send(client)
    }
}

/// A sampling contract field as the proto carries it.
pub fn sampling_value(params: &proto::SamplingParams, field: SamplingField) -> Option<f64> {
    // The decimal a JSON caller would send: 0.7, not the f32's 0.699999988.
    let float = |value: Option<f32>| value.map(|v| v.to_string().parse().unwrap_or(f64::from(v)));
    match field {
        SamplingField::Temperature => float(params.temperature),
        SamplingField::TopP => float(params.top_p),
        SamplingField::MinP => float(params.min_p),
        SamplingField::RepetitionPenalty => float(params.repetition_penalty),
        SamplingField::FrequencyPenalty => float(params.frequency_penalty),
        SamplingField::PresencePenalty => float(params.presence_penalty),
        SamplingField::TopK => params.top_k.map(f64::from),
        SamplingField::N => params.n.map(f64::from),
    }
}

fn set_sampling_default(params: &mut proto::SamplingParams, field: SamplingField, value: f64) {
    let float = Some(value as f32);
    let slot = match field {
        SamplingField::Temperature => &mut params.temperature,
        SamplingField::TopP => &mut params.top_p,
        SamplingField::MinP => &mut params.min_p,
        SamplingField::RepetitionPenalty => &mut params.repetition_penalty,
        SamplingField::FrequencyPenalty => &mut params.frequency_penalty,
        SamplingField::PresencePenalty => &mut params.presence_penalty,
        SamplingField::TopK => return params.top_k = params.top_k.or(Some(value as i32)),
        SamplingField::N => return params.n = params.n.or(Some(value as i32)),
    };
    *slot = slot.or(float);
}

/// Apply `/generate`'s additions to a typed generate request's fields.
fn apply_generation(
    additions: &Additions,
    rid: &mut Option<String>,
    routed_dp_rank: &mut Option<i32>,
    disaggregated: &mut Option<proto::DisaggregatedParams>,
    params: &mut Option<proto::SamplingParams>,
) {
    if rid.is_none() {
        rid.clone_from(&additions.rid);
    }
    if let Some(rank) = additions.routed_dp_rank {
        *routed_dp_rank = rank.and_then(|rank| i32::try_from(rank).ok());
    }
    if additions.bootstrap.is_some() {
        disaggregated.clone_from(&additions.bootstrap);
    }
    if !additions.sampling_defaults.is_empty() {
        let params = params.get_or_insert_with(Default::default);
        for &(field, value) in &additions.sampling_defaults {
            set_sampling_default(params, field, value);
        }
    }
}

impl TypedRpc for proto::GenerateRequest {
    type Reply = proto::GenerateResponse;

    fn view(&self) -> RoutingView<'_> {
        RoutingView {
            input_ids: &self.input_ids,
            text: None,
            sampling_params: self.sampling_params.as_ref(),
            stream: self.stream.unwrap_or(false),
            rid: self.rid.as_deref(),
        }
    }

    fn apply(&mut self, additions: &Additions) {
        apply_generation(
            additions,
            &mut self.rid,
            &mut self.routed_dp_rank,
            &mut self.disaggregated_params,
            &mut self.sampling_params,
        );
    }

    fn trace_headers(&mut self) -> &mut HashMap<String, String> {
        &mut self.trace_headers
    }

    fn send(self, mut client: SglangServiceClient<Channel>) -> RpcFuture<Self::Reply> {
        Box::pin(async move { Ok(client.generate(self).await?.into_inner().boxed()) })
    }
}

impl TypedRpc for proto::TextGenerateRequest {
    type Reply = proto::TextGenerateResponse;

    fn view(&self) -> RoutingView<'_> {
        RoutingView {
            input_ids: &[],
            text: Some(&self.text),
            sampling_params: self.sampling_params.as_ref(),
            stream: self.stream.unwrap_or(false),
            rid: self.rid.as_deref(),
        }
    }

    fn apply(&mut self, additions: &Additions) {
        apply_generation(
            additions,
            &mut self.rid,
            &mut self.routed_dp_rank,
            &mut self.disaggregated_params,
            &mut self.sampling_params,
        );
    }

    fn trace_headers(&mut self) -> &mut HashMap<String, String> {
        &mut self.trace_headers
    }

    fn send(self, mut client: SglangServiceClient<Channel>) -> RpcFuture<Self::Reply> {
        Box::pin(async move { Ok(client.text_generate(self).await?.into_inner().boxed()) })
    }
}

/// `TextGenerate` as `Generate` with the router's tokens, asking for the text back.
/// Text logprobs are lost, so a request asking for them stays `TextGenerate`.
pub fn text_to_generate(
    request: proto::TextGenerateRequest,
    input_ids: Vec<i32>,
) -> proto::GenerateRequest {
    let proto::TextGenerateRequest {
        text: _,
        sampling_params,
        stream,
        return_logprob,
        top_logprobs_num,
        logprob_start_len,
        return_text_in_logprobs: _,
        rid,
        lora_path,
        routing_key,
        routed_dp_rank,
        trace_headers,
        session_id,
        disaggregated_params,
        priority,
        require_reasoning,
        max_thinking_tokens,
        kv_hints,
    } = request;
    proto::GenerateRequest {
        input_ids,
        sampling_params,
        stream,
        return_logprob,
        top_logprobs_num,
        logprob_start_len,
        rid,
        lora_path,
        routing_key,
        routed_dp_rank,
        trace_headers,
        session_id,
        disaggregated_params,
        priority,
        require_reasoning,
        max_thinking_tokens,
        kv_hints,
        return_text: Some(true),
    }
}

/// The reply a `TextGenerate` caller expects, from `Generate` with `return_text`.
pub fn generate_to_text(reply: proto::GenerateResponse) -> proto::TextGenerateResponse {
    proto::TextGenerateResponse {
        text: reply.text.unwrap_or_default(),
        meta_info: reply.meta_info,
        finished: reply.finished,
    }
}

/// Unary typed RPCs routed like embeddings: only the rid is added.
macro_rules! embedding_rpc {
    ($request:ident -> $reply:ident, $method:ident, |$this:ident| $input_ids:expr, $text:expr) => {
        impl TypedRpc for proto::$request {
            type Reply = proto::$reply;

            fn view(&self) -> RoutingView<'_> {
                let $this = self;
                RoutingView {
                    input_ids: $input_ids,
                    text: $text,
                    sampling_params: None,
                    stream: false,
                    rid: self.rid.as_deref(),
                }
            }

            fn apply(&mut self, additions: &Additions) {
                if self.rid.is_none() {
                    self.rid.clone_from(&additions.rid);
                }
            }

            fn trace_headers(&mut self) -> &mut HashMap<String, String> {
                &mut self.trace_headers
            }

            fn send(self, mut client: SglangServiceClient<Channel>) -> RpcFuture<Self::Reply> {
                Box::pin(async move {
                    let reply = client.$method(self).await?.into_inner();
                    Ok(futures::stream::iter([Ok(reply)]).boxed())
                })
            }
        }
    };
}

embedding_rpc!(EmbedRequest -> EmbedResponse, embed, |r| &r.input_ids, None);
embedding_rpc!(TextEmbedRequest -> TextEmbedResponse, text_embed, |r| &[], Some(&r.text));
embedding_rpc!(ClassifyRequest -> ClassifyResponse, classify, |r| &r.input_ids,
    Some(r.text.as_str()).filter(|_| r.input_ids.is_empty()));

/// `TextEmbed` as `Embed` with the router's tokens; both reply with an embedding.
pub fn text_to_embed(request: proto::TextEmbedRequest, input_ids: Vec<i32>) -> proto::EmbedRequest {
    proto::EmbedRequest {
        input_ids,
        rid: request.rid,
        routing_key: request.routing_key,
        trace_headers: request.trace_headers,
    }
}

pub fn embed_to_text(reply: proto::EmbedResponse) -> proto::TextEmbedResponse {
    proto::TextEmbedResponse {
        embedding: reply.embedding,
        meta_info: reply.meta_info,
    }
}

/// A failed call's effect on the breaker; only faults count toward opening it.
fn status_breaker_outcome(code: Code) -> BreakerOutcome {
    match code {
        Code::ResourceExhausted => BreakerOutcome::Neutral,
        Code::Unavailable
        | Code::Internal
        | Code::Unknown
        | Code::DeadlineExceeded
        | Code::DataLoss => BreakerOutcome::Failure,
        _ => BreakerOutcome::Success,
    }
}

/// The router gave up waiting on the engine: HTTP's 504 `upstream_timeout`.
fn upstream_timeout() -> Status {
    router_grpc_status(
        Code::DeadlineExceeded,
        "upstream_timeout",
        "upstream request timed out",
    )
}

/// The engine's own status when it ended the stream; the router's own endings
/// keep the HTTP path's error codes.
fn into_status(error: std::io::Error) -> Status {
    let Some(inner) = error.into_inner() else {
        return Status::unavailable("stream ended without a result");
    };
    let inner = match inner.downcast::<Status>() {
        Ok(status) => return *status,
        Err(inner) => inner,
    };
    let Ok(router) = inner.downcast::<RouterStreamError>() else {
        return Status::unavailable("upstream stream failed");
    };
    let (code, error_code, message) = match router.reason {
        StreamEndReason::IdleTimeout => return upstream_timeout(),
        StreamEndReason::Expired => (
            Code::DeadlineExceeded,
            "stale_request_expired",
            "request expired before completion",
        ),
        StreamEndReason::Aborted => (
            Code::Unavailable,
            "prefill_failed",
            "prefill failed before KV transfer completed",
        ),
        _ => (Code::Internal, "internal_error", "internal error"),
    };
    router_grpc_status(code, error_code, message)
}

/// Aborts the engine request by the router's rid unless disarmed, as
/// `/abort_request` does over HTTP. A dropped call alone is not enough: the
/// engine's OpenAI RPCs abort an id of their own, not the request's rid.
struct AbortOnDrop(Option<(SglangServiceClient<Channel>, String)>);

impl AbortOnDrop {
    fn new(client: &SglangServiceClient<Channel>, rid: Option<&str>) -> Self {
        let rid = rid.filter(|rid| !rid.is_empty());
        Self(rid.map(|rid| (client.clone(), rid.to_owned())))
    }

    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        let Some((mut client, rid)) = self.0.take() else {
            return;
        };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let abort = client.abort(proto::AbortRequest {
                    rid,
                    abort_all: false,
                });
                match tokio::time::timeout(Duration::from_secs(5), abort).await {
                    Ok(Ok(_)) => {}
                    Ok(Err(error)) => tracing::warn!(%error, "engine abort failed"),
                    Err(_) => tracing::warn!("engine abort timed out"),
                }
            });
        }
    }
}

/// The headers the HTTP path forwards; the engine reads these as request headers.
fn trace_headers(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter(|(name, _)| should_forward_request_header(name))
        .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_owned())))
        .collect()
}

impl Proxy {
    /// Call `rpc` on `worker`. A streaming call goes through the same pump as
    /// SSE; any other is read to the end within the request timeout. An
    /// unfinished call aborts the engine request `abort_rid`.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_grpc<R: EngineRpc>(
        &self,
        worker: &Worker,
        rpc: &R,
        headers: &HeaderMap,
        body: Bytes,
        additions: &Additions,
        abort_rid: Option<&str>,
        streaming: bool,
        stream_guards: Option<Box<dyn Send + 'static>>,
        on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
        on_stream_end: Option<Box<dyn FnOnce(StreamEnd) + Send + 'static>>,
        expiration: Option<CancellationToken>,
        stream_abort: Option<CancellationToken>,
    ) -> Result<GrpcResponse<R::Reply>, ApiError> {
        let client = worker.grpc().ok_or_else(|| ApiError::WorkerMisconfigured {
            worker: worker.url.clone(),
            source: anyhow::anyhow!("the engine reports no --grpc-port"),
        })?;
        let breaker = &worker.breaker;
        let permit = breaker.acquire().ok_or_else(|| ApiError::BreakerOpen {
            worker: worker.url.clone(),
        })?;
        let mut abort = AbortOnDrop::new(&client, abort_rid);
        let call = rpc.call(client, body, additions, trace_headers(headers));
        let failed = |status: &Status| status_breaker_outcome(status.code()).record(breaker);
        let respond = |status, replies: Vec<_>, mut abort: AbortOnDrop| {
            breaker_outcome(status).record(breaker);
            abort.disarm();
            GrpcResponse {
                status,
                replies: futures::stream::iter(replies.into_iter().map(Ok)).boxed(),
            }
        };
        if !streaming {
            let collect = async { call.await?.collect::<Vec<_>>().await.into_iter().collect() };
            let replies: Vec<_> = tokio::time::timeout(self.request_timeout, collect)
                .await
                .unwrap_or_else(|_| Err(upstream_timeout()))
                .inspect_err(failed)
                .map_err(ApiError::UpstreamGrpc)?;
            let status = replies.last().and_then(R::status).unwrap_or(StatusCode::OK);
            permit.disarm();
            drop(stream_guards);
            return Ok(respond(status, replies, abort));
        }
        let call = async {
            let mut stream = call.await?;
            let first = stream.next().await.transpose()?;
            Ok::<_, Status>((first, stream))
        };
        // The engine answers a request it rejects with one reply carrying an HTTP
        // status, as an HTTP engine answers with a status line.
        let idle = self.stream_idle_timeout.unwrap_or(Duration::MAX);
        let (first, stream) = match tokio::time::timeout(idle, call).await {
            Ok(result) => result.inspect_err(failed).map_err(ApiError::UpstreamGrpc)?,
            Err(_) => {
                breaker.record_failure();
                return Err(ApiError::UpstreamGrpc(upstream_timeout()));
            }
        };
        if let Some(status) = first.as_ref().and_then(R::status) {
            permit.disarm();
            drop(stream_guards);
            return Ok(respond(status, first.into_iter().collect(), abort));
        }
        // A status the engine ends the stream with decides the breaker outcome.
        let ended_with = Arc::new(OnceLock::new());
        let record = Arc::clone(&ended_with);
        let stream = futures::stream::iter(first.map(Ok))
            .chain(stream.inspect_err(move |status| _ = record.set(status.code())));
        let breaker = Arc::clone(breaker);
        let on_complete: Box<dyn FnOnce(StreamEnd) + Send + 'static> = Box::new(move |end| {
            let outcome = match ended_with.get() {
                Some(&code) if end.reason == StreamEndReason::UpstreamError => {
                    status_breaker_outcome(code)
                }
                _ => stream_breaker_outcome(end),
            };
            outcome.record(&breaker);
            if end.reason == StreamEndReason::Completed {
                abort.disarm();
            }
            if let Some(hook) = on_stream_end {
                hook(end);
            }
        });
        permit.disarm();
        let replies = sse::pump(
            stream,
            stream_guards,
            Some(on_complete),
            on_first_byte,
            StreamLimits {
                idle_timeout: self.stream_idle_timeout,
                expiration,
                abort: stream_abort,
            },
        );
        Ok(GrpcResponse {
            status: StatusCode::OK,
            replies: replies.map(|reply| reply.map_err(into_status)).boxed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_values_read_as_the_decimal_sent() {
        let params = proto::SamplingParams {
            temperature: Some(0.7),
            top_k: Some(20),
            ..Default::default()
        };
        assert_eq!(
            sampling_value(&params, SamplingField::Temperature),
            Some(0.7)
        );
        assert_eq!(sampling_value(&params, SamplingField::TopK), Some(20.0));
    }

    #[test]
    fn apply_adds_router_fields_without_overriding_the_caller() {
        let mut request = proto::GenerateRequest {
            rid: Some("caller".into()),
            routed_dp_rank: Some(3),
            sampling_params: Some(proto::SamplingParams {
                temperature: Some(0.5),
                ..Default::default()
            }),
            ..Default::default()
        };
        let room = proto::DisaggregatedParams {
            bootstrap_host: "p".into(),
            bootstrap_port: 8998,
            bootstrap_room: 42,
        };
        request.apply(&Additions {
            rid: Some("router".into()),
            bootstrap: Some(room.clone()),
            sampling_defaults: vec![
                (SamplingField::Temperature, 1.0),
                (SamplingField::TopK, 20.0),
            ],
            routed_dp_rank: Some(None),
        });

        assert_eq!(request.rid.as_deref(), Some("caller"));
        assert_eq!(
            request.routed_dp_rank, None,
            "the router's unpinned rank wins"
        );
        let params = request.sampling_params.unwrap();
        assert_eq!((params.temperature, params.top_k), (Some(0.5), Some(20)));
        assert_eq!(request.disaggregated_params, Some(room));
    }
}
