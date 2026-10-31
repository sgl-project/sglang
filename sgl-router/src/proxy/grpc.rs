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
use futures::stream::{BoxStream, StreamExt, TryStreamExt};
use sglang_api_types::api::v1 as proto;
use sglang_api_types::api::v1::sglang_service_client::SglangServiceClient;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;
use tonic::{Code, Status};

pub type Replies = BoxStream<'static, Result<proto::GenerateStreamItem, Status>>;

/// An engine's answer to a gRPC call; a unary reply is a one-item stream.
pub struct GrpcResponse {
    /// Status used by routing for an error item; successful frames use 200.
    pub status: StatusCode,
    pub replies: Replies,
}

impl PumpItem for proto::GenerateStreamItem {
    fn is_error_event(&self, _: &mut ErrorEventScanner) -> bool {
        response_status(self).is_some()
    }
}

/// An item error or a scheduler abort carrying an HTTP error status.
fn response_status(reply: &proto::GenerateStreamItem) -> Option<StatusCode> {
    use proto::generate_stream_item::Item;
    let code = match reply.item.as_ref()? {
        Item::Error(error) => error.error.as_ref().map(|error| error.code),
        Item::Frame(frame) => {
            let reason = frame.meta_info.as_ref()?.finish_reason.as_ref()?;
            match reason.kind.as_ref()? {
                proto::finish_reason::Kind::Abort(abort) => Some(abort.status_code?),
                _ => return None,
            }
        }
    };
    Some(
        code.and_then(|code| u16::try_from(code).ok())
            .filter(|code| (400..600).contains(code))
            .and_then(|code| StatusCode::from_u16(code).ok())
            .unwrap_or(StatusCode::BAD_GATEWAY),
    )
}

/// The single prompt and sampling fields used by routing.
pub struct RoutingView<'a> {
    pub input_ids: Option<&'a [i64]>,
    pub text: Option<&'a str>,
    pub sampling_params: Option<&'a proto::SamplingParams>,
    pub stream: bool,
    pub rid: Option<&'a str>,
}

impl<'a> TryFrom<&'a proto::GenerateRequest> for RoutingView<'a> {
    type Error = ApiError;

    fn try_from(request: &'a proto::GenerateRequest) -> Result<Self, Self::Error> {
        let batch = |field| {
            ApiError::BadRequest(format!(
                "batch {field} is not supported by sgl-router gRPC Generate"
            ))
        };
        let text = match request.text.as_ref().and_then(|text| text.value.as_ref()) {
            Some(proto::string_or_list::Value::One(text)) => Some(text.as_str()),
            Some(proto::string_or_list::Value::Many(_)) => return Err(batch("text")),
            None => None,
        };
        let input_ids = match request
            .input_ids
            .as_ref()
            .and_then(|ids| ids.value.as_ref())
        {
            Some(proto::token_ids_or_list::Value::One(ids)) => Some(ids.ids.as_slice()),
            Some(proto::token_ids_or_list::Value::Many(_)) => return Err(batch("input_ids")),
            None => None,
        };
        if text.is_some() == input_ids.is_some() {
            return Err(ApiError::BadRequest(
                "provide exactly one of text or input_ids".into(),
            ));
        }
        let rid = match request.rid.as_ref().and_then(|rid| rid.value.as_ref()) {
            Some(proto::string_or_list::Value::One(rid)) => Some(rid.as_str()),
            Some(proto::string_or_list::Value::Many(_)) => return Err(batch("rid")),
            None => None,
        };
        let sampling_params = match request
            .sampling_params
            .as_ref()
            .and_then(|params| params.value.as_ref())
        {
            Some(proto::sampling_params_or_list::Value::One(params)) => Some(params),
            Some(proto::sampling_params_or_list::Value::Many(_)) => {
                return Err(batch("sampling_params"))
            }
            None => None,
        };
        Ok(Self {
            input_ids,
            text,
            sampling_params,
            stream: request.stream.unwrap_or(false),
            rid,
        })
    }
}

/// What the router adds to a typed request, as it adds it to an HTTP body.
#[derive(Clone, Default)]
pub struct Additions {
    pub rid: Option<String>,
    pub bootstrap: Option<BootstrapFields>,
    pub sampling_defaults: Vec<(SamplingField, f64)>,
    /// Set when the router owns the DP rank; `Some(None)` unpins it.
    pub routed_dp_rank: Option<Option<u32>>,
    pub disagg_prefill_dp_rank: Option<u32>,
}

/// The PD rendezvous shared by the prefill and decode requests.
#[derive(Clone)]
pub struct BootstrapFields {
    pub host: String,
    pub port: Option<u16>,
    pub room: u64,
}

/// A sampling contract field as the proto carries it.
pub fn sampling_value(params: &proto::SamplingParams, field: SamplingField) -> Option<f64> {
    match field {
        SamplingField::Temperature => params.temperature,
        SamplingField::TopP => params.top_p,
        SamplingField::MinP => params.min_p,
        SamplingField::RepetitionPenalty => params.repetition_penalty,
        SamplingField::FrequencyPenalty => params.frequency_penalty,
        SamplingField::PresencePenalty => params.presence_penalty,
        SamplingField::TopK => params.top_k.map(|v| v as f64),
        SamplingField::N => params.n.map(|v| v as f64),
    }
}

fn set_sampling_default(params: &mut proto::SamplingParams, field: SamplingField, value: f64) {
    let slot = match field {
        SamplingField::Temperature => &mut params.temperature,
        SamplingField::TopP => &mut params.top_p,
        SamplingField::MinP => &mut params.min_p,
        SamplingField::RepetitionPenalty => &mut params.repetition_penalty,
        SamplingField::FrequencyPenalty => &mut params.frequency_penalty,
        SamplingField::PresencePenalty => &mut params.presence_penalty,
        SamplingField::TopK => return params.top_k = params.top_k.or(Some(value as i64)),
        SamplingField::N => return params.n = params.n.or(Some(value as i64)),
    };
    *slot = slot.or(Some(value));
}

fn apply_generation(request: &mut proto::GenerateRequest, additions: &Additions) {
    if request
        .rid
        .as_ref()
        .and_then(|rid| rid.value.as_ref())
        .is_none()
    {
        request.rid = additions.rid.as_ref().map(|rid| proto::StringOrList {
            value: Some(proto::string_or_list::Value::One(rid.clone())),
        });
    }
    if let Some(rank) = additions.routed_dp_rank {
        request.routed_dp_rank = rank.map(i64::from);
        request.disagg_prefill_dp_rank = additions.disagg_prefill_dp_rank.map(i64::from);
    }
    if let Some(bootstrap) = &additions.bootstrap {
        request.bootstrap_host = Some(proto::OptionalStringOrList {
            value: Some(proto::optional_string_or_list::Value::One(
                proto::OptionalString {
                    value: Some(bootstrap.host.clone()),
                },
            )),
        });
        let one = |value| {
            Some(proto::OptionalInt64OrList {
                value: Some(proto::optional_int64_or_list::Value::One(
                    proto::OptionalInt64 { value },
                )),
            })
        };
        request.bootstrap_port = one(bootstrap.port.map(i64::from));
        // Router-generated rooms are bounded to signed int64.
        request.bootstrap_room = one(Some(bootstrap.room as i64));
    }
    if !additions.sampling_defaults.is_empty() {
        let params = request.sampling_params.get_or_insert_with(Default::default);
        let params = params
            .value
            .get_or_insert_with(|| proto::sampling_params_or_list::Value::One(Default::default()));
        if let proto::sampling_params_or_list::Value::One(params) = params {
            for &(field, value) in &additions.sampling_defaults {
                set_sampling_default(params, field, value);
            }
        }
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
/// `/abort_request` does over HTTP.
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

impl Proxy {
    /// Send Generate to `worker`. A streaming call goes through the same pump as
    /// SSE; any other is read to the end within the request timeout. An
    /// unfinished call aborts the engine request `abort_rid`.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_grpc(
        &self,
        worker: &Worker,
        request: &proto::GenerateRequest,
        headers: &HeaderMap,
        additions: &Additions,
        abort_rid: Option<&str>,
        streaming: bool,
        stream_guards: Option<Box<dyn Send + 'static>>,
        on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
        on_stream_end: Option<Box<dyn FnOnce(StreamEnd) + Send + 'static>>,
        expiration: Option<CancellationToken>,
        stream_abort: Option<CancellationToken>,
    ) -> Result<GrpcResponse, ApiError> {
        let mut client = worker.grpc().ok_or_else(|| ApiError::WorkerMisconfigured {
            worker: worker.url.clone(),
            source: anyhow::anyhow!("the engine reports no --grpc-port"),
        })?;
        let breaker = &worker.breaker;
        let permit = breaker.acquire().ok_or_else(|| ApiError::BreakerOpen {
            worker: worker.url.clone(),
        })?;
        let mut abort = AbortOnDrop::new(&client, abort_rid);
        let mut request = request.clone();
        apply_generation(&mut request, additions);
        let mut request = tonic::Request::new(request);
        let mut forwarded = HeaderMap::new();
        for (name, value) in headers {
            if name != "authorization" && should_forward_request_header(name) {
                forwarded.append(name.clone(), value.clone());
            }
        }
        *request.metadata_mut() = tonic::metadata::MetadataMap::from_headers(forwarded);
        // Keep the tonic future off the nested routing/dispatch futures' stacks.
        let call =
            Box::pin(async move { Ok::<_, Status>(client.generate(request).await?.into_inner()) });
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
            let status = replies
                .last()
                .and_then(response_status)
                .unwrap_or(StatusCode::OK);
            permit.disarm();
            drop(stream_guards);
            return Ok(respond(status, replies, abort));
        }
        let call = async {
            let mut stream = call.await?;
            let first = stream.next().await.transpose()?;
            Ok::<_, Status>((first, stream))
        };
        // A first error item can still be retried before any frame reaches the client.
        let idle = self.stream_idle_timeout.unwrap_or(Duration::MAX);
        let (first, stream) = match tokio::time::timeout(idle, call).await {
            Ok(result) => result.inspect_err(failed).map_err(ApiError::UpstreamGrpc)?,
            Err(_) => {
                breaker.record_failure();
                return Err(ApiError::UpstreamGrpc(upstream_timeout()));
            }
        };
        if let Some(status) = first.as_ref().and_then(response_status) {
            permit.disarm();
            drop(stream_guards);
            return Ok(respond(status, first.into_iter().collect(), abort));
        }
        // A status the engine ends the stream with decides the breaker outcome.
        let ended_with = Arc::new(OnceLock::new());
        let record = Arc::clone(&ended_with);
        let item_status = Arc::new(OnceLock::new());
        let record_item = Arc::clone(&item_status);
        let stream = futures::stream::iter(first.map(Ok))
            .chain(stream.inspect_err(move |status| _ = record.set(status.code())))
            .inspect_ok(move |item| {
                if let Some(status) = response_status(item) {
                    let _ = record_item.set(status);
                }
            });
        let breaker = Arc::clone(breaker);
        let on_complete: Box<dyn FnOnce(StreamEnd) + Send + 'static> = Box::new(move |end| {
            let outcome = match ended_with.get() {
                Some(&code) if end.reason == StreamEndReason::UpstreamError => {
                    status_breaker_outcome(code)
                }
                _ if end.reason == StreamEndReason::Completed => item_status
                    .get()
                    .copied()
                    .map(breaker_outcome)
                    .unwrap_or_else(|| stream_breaker_outcome(end)),
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
    fn apply_adds_router_fields_without_overriding_the_caller() {
        for caller_params in [false, true] {
            let rid = proto::StringOrList {
                value: Some(proto::string_or_list::Value::One("caller".into())),
            };
            let mut request = proto::GenerateRequest {
                rid: Some(rid.clone()),
                routed_dp_rank: Some(3),
                sampling_params: caller_params.then(|| proto::SamplingParamsOrList {
                    value: Some(proto::sampling_params_or_list::Value::One(
                        proto::SamplingParams {
                            temperature: Some(0.70000000001),
                            ..Default::default()
                        },
                    )),
                }),
                ..Default::default()
            };
            apply_generation(
                &mut request,
                &Additions {
                    rid: Some("router".into()),
                    sampling_defaults: vec![
                        (SamplingField::Temperature, 1.0),
                        (SamplingField::TopK, 20.0),
                    ],
                    routed_dp_rank: Some(None),
                    ..Default::default()
                },
            );
            assert_eq!(request.rid, Some(rid));
            assert_eq!(
                request.routed_dp_rank, None,
                "the router's unpinned rank wins"
            );
            let proto::sampling_params_or_list::Value::One(params) =
                request.sampling_params.unwrap().value.unwrap()
            else {
                panic!("expected scalar sampling params");
            };
            assert_eq!(
                params,
                proto::SamplingParams {
                    temperature: Some(if caller_params { 0.70000000001 } else { 1.0 }),
                    top_k: Some(20),
                    ..Default::default()
                }
            );
        }
    }
}
