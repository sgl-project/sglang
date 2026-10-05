// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Forwarding to an engine's native gRPC `SglangService`, for gRPC clients.

use super::sse::{self, ErrorEventScanner, PumpItem, StreamEnd, StreamLimits};
use super::{breaker_outcome, stream_breaker_outcome, BreakerOutcome, Proxy};
use crate::config::SamplingField;
use crate::server::error::ApiError;
use crate::server::header_utils::should_forward_request_header;
use crate::workers::Worker;
use axum::http::{HeaderMap, StatusCode};
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, StreamExt};
use serde_json::{json, Map, Value};
use sglang_grpc_types::sglang::runtime::v1 as proto;
use sglang_grpc_types::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use std::collections::HashMap;
use std::sync::Arc;
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

    /// Call the engine; typed requests take the router's additions from `body`.
    fn call(
        &self,
        client: SglangServiceClient<Channel>,
        body: Bytes,
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

/// A typed RPC, routed through the JSON its HTTP sibling reads.
pub trait TypedRpc: Clone + Send + Sync + 'static {
    type Reply: PumpItem;

    /// The body its HTTP sibling would receive, for the router's preparation.
    fn view(&self) -> Value;

    /// Take what the router added to the prepared body: rid, PD bootstrap,
    /// DP rank and sampling defaults. Fields the proto cannot carry are dropped.
    fn patch(&mut self, prepared: &Map<String, Value>);

    fn send(self, client: SglangServiceClient<Channel>) -> RpcFuture<Self::Reply>;
}

type RpcFuture<T> = BoxFuture<'static, Result<Replies<T>, Status>>;

impl<T: TypedRpc> EngineRpc for T {
    type Reply = T::Reply;

    fn call(
        &self,
        client: SglangServiceClient<Channel>,
        body: Bytes,
        _: HashMap<String, String>,
    ) -> RpcFuture<T::Reply> {
        let mut request = self.clone();
        match serde_json::from_slice(&body) {
            Ok(Value::Object(prepared)) => request.patch(&prepared),
            _ => return Box::pin(async { Err(Status::internal("unparsable prepared body")) }),
        }
        request.send(client)
    }
}

/// `sampling_params` as `/generate` reads them: the contract's fields and the output budget.
fn sampling_view(params: &Option<proto::SamplingParams>) -> Value {
    let Some(p) = params else {
        return Value::Null;
    };
    json!({
        "temperature": p.temperature,
        "top_p": p.top_p,
        "top_k": p.top_k,
        "min_p": p.min_p,
        "repetition_penalty": p.repetition_penalty,
        "frequency_penalty": p.frequency_penalty,
        "presence_penalty": p.presence_penalty,
        "n": p.n,
        "max_new_tokens": p.max_new_tokens,
    })
}

/// The prepared `rid`, unless the caller set one.
fn patch_rid(rid: &mut Option<String>, prepared: &Map<String, Value>) {
    if rid.is_none() {
        *rid = prepared
            .get("rid")
            .and_then(Value::as_str)
            .map(str::to_owned);
    }
}

/// Copy `/generate`'s additions onto a typed generate request.
fn patch_generation(
    prepared: &Map<String, Value>,
    rid: &mut Option<String>,
    routed_dp_rank: &mut Option<i32>,
    disaggregated: &mut Option<proto::DisaggregatedParams>,
    params: &mut Option<proto::SamplingParams>,
) {
    patch_rid(rid, prepared);
    // The router owns the DP rank when it sets one, as over HTTP; null unpins it.
    if let Some(rank) = prepared.get("routed_dp_rank") {
        *routed_dp_rank = rank.as_i64().and_then(|rank| i32::try_from(rank).ok());
    }
    if let Some(room) = prepared.get("bootstrap_room").and_then(Value::as_i64) {
        *disaggregated = Some(proto::DisaggregatedParams {
            bootstrap_host: prepared["bootstrap_host"]
                .as_str()
                .unwrap_or_default()
                .into(),
            bootstrap_port: prepared["bootstrap_port"].as_i64().unwrap_or_default() as i32,
            bootstrap_room: room,
        });
    }
    let defaults = &prepared.get("sampling_params").unwrap_or(&Value::Null);
    let params = params.get_or_insert_with(Default::default);
    for field in SamplingField::ALL {
        let Some(value) = defaults[field.wire_name()].as_f64() else {
            continue;
        };
        let (float, int) = (Some(value as f32), Some(value as i32));
        let slot = match field {
            SamplingField::Temperature => &mut params.temperature,
            SamplingField::TopP => &mut params.top_p,
            SamplingField::MinP => &mut params.min_p,
            SamplingField::RepetitionPenalty => &mut params.repetition_penalty,
            SamplingField::FrequencyPenalty => &mut params.frequency_penalty,
            SamplingField::PresencePenalty => &mut params.presence_penalty,
            SamplingField::TopK => {
                params.top_k = params.top_k.or(int);
                continue;
            }
            SamplingField::N => {
                params.n = params.n.or(int);
                continue;
            }
        };
        *slot = slot.or(float);
    }
}

impl TypedRpc for proto::GenerateRequest {
    type Reply = proto::GenerateResponse;

    fn view(&self) -> Value {
        json!({
            "input_ids": self.input_ids,
            "sampling_params": sampling_view(&self.sampling_params),
            "stream": self.stream,
            "rid": self.rid,
        })
    }

    fn patch(&mut self, prepared: &Map<String, Value>) {
        patch_generation(
            prepared,
            &mut self.rid,
            &mut self.routed_dp_rank,
            &mut self.disaggregated_params,
            &mut self.sampling_params,
        );
    }

    fn send(self, mut client: SglangServiceClient<Channel>) -> RpcFuture<Self::Reply> {
        Box::pin(async move { Ok(client.generate(self).await?.into_inner().boxed()) })
    }
}

impl TypedRpc for proto::TextGenerateRequest {
    type Reply = proto::TextGenerateResponse;

    fn view(&self) -> Value {
        json!({
            "text": self.text,
            "sampling_params": sampling_view(&self.sampling_params),
            "stream": self.stream,
            "rid": self.rid,
        })
    }

    fn patch(&mut self, prepared: &Map<String, Value>) {
        patch_generation(
            prepared,
            &mut self.rid,
            &mut self.routed_dp_rank,
            &mut self.disaggregated_params,
            &mut self.sampling_params,
        );
    }

    fn send(self, mut client: SglangServiceClient<Channel>) -> RpcFuture<Self::Reply> {
        Box::pin(async move { Ok(client.text_generate(self).await?.into_inner().boxed()) })
    }
}

/// Unary typed RPCs routed like embeddings: the view is an `input` and only the rid comes back.
macro_rules! embedding_rpc {
    ($request:ident -> $reply:ident, $method:ident, |$this:ident| $input:expr) => {
        impl TypedRpc for proto::$request {
            type Reply = proto::$reply;

            fn view(&self) -> Value {
                let $this = self;
                json!({ "input": $input, "rid": self.rid })
            }

            fn patch(&mut self, prepared: &Map<String, Value>) {
                patch_rid(&mut self.rid, prepared);
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

embedding_rpc!(EmbedRequest -> EmbedResponse, embed, |r| r.input_ids);
embedding_rpc!(TextEmbedRequest -> TextEmbedResponse, text_embed, |r| r.text);
embedding_rpc!(ClassifyRequest -> ClassifyResponse, classify, |r| if r.input_ids.is_empty() {
    json!(r.text)
} else {
    json!(r.input_ids)
});

/// A failed call's effect on the breaker; only faults count toward opening it.
fn status_breaker_outcome(status: &Status) -> BreakerOutcome {
    match status.code() {
        Code::ResourceExhausted => BreakerOutcome::Neutral,
        Code::Unavailable
        | Code::Internal
        | Code::Unknown
        | Code::DeadlineExceeded
        | Code::DataLoss => BreakerOutcome::Failure,
        _ => BreakerOutcome::Success,
    }
}

/// The engine's own status when it ended the stream, else the router's reason.
fn into_status(error: std::io::Error) -> Status {
    match error.into_inner().map(|inner| inner.downcast::<Status>()) {
        Some(Ok(status)) => *status,
        Some(Err(inner)) => Status::unavailable(inner.to_string()),
        None => Status::unavailable("stream ended without a result"),
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
    /// SSE; any other is read to the end within the request timeout.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_grpc<R: EngineRpc>(
        &self,
        worker: &Worker,
        rpc: &R,
        headers: &HeaderMap,
        body: Bytes,
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
        let call = rpc.call(client, body, trace_headers(headers));
        if !streaming {
            let collect = async { call.await?.collect::<Vec<_>>().await.into_iter().collect() };
            let replies: Vec<_> = tokio::time::timeout(self.request_timeout, collect)
                .await
                .unwrap_or_else(|_| Err(Status::deadline_exceeded("upstream request timed out")))
                .inspect_err(|status| status_breaker_outcome(status).record(breaker))
                .map_err(ApiError::UpstreamGrpc)?;
            let status = replies.last().and_then(R::status).unwrap_or(StatusCode::OK);
            breaker_outcome(status).record(breaker);
            permit.disarm();
            drop(stream_guards);
            return Ok(GrpcResponse {
                status,
                replies: futures::stream::iter(replies.into_iter().map(Ok)).boxed(),
            });
        }
        let stream = call
            .await
            .inspect_err(|status| status_breaker_outcome(status).record(breaker))
            .map_err(ApiError::UpstreamGrpc)?;
        let breaker = Arc::clone(breaker);
        let on_complete: Box<dyn FnOnce(StreamEnd) + Send + 'static> = Box::new(move |end| {
            stream_breaker_outcome(end).record(&breaker);
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
    fn patch_takes_router_additions_without_overriding_the_caller() {
        let mut request = proto::GenerateRequest {
            rid: Some("caller".into()),
            routed_dp_rank: Some(3),
            sampling_params: Some(proto::SamplingParams {
                temperature: Some(0.5),
                ..Default::default()
            }),
            ..Default::default()
        };
        let prepared = json!({
            "rid": "router",
            "routed_dp_rank": null,
            "bootstrap_host": "p", "bootstrap_port": 8998, "bootstrap_room": 42,
            "sampling_params": {"temperature": 1.0, "top_k": 20},
        });
        request.patch(prepared.as_object().unwrap());

        assert_eq!(request.rid.as_deref(), Some("caller"));
        assert_eq!(request.routed_dp_rank, None, "a null rank unpins");
        let params = request.sampling_params.unwrap();
        assert_eq!((params.temperature, params.top_k), (Some(0.5), Some(20)));
        let room = request.disaggregated_params.unwrap();
        assert_eq!(
            (
                room.bootstrap_host.as_str(),
                room.bootstrap_port,
                room.bootstrap_room
            ),
            ("p", 8998, 42)
        );
    }
}
