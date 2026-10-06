// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Forwarding to an engine's native gRPC `SglangService`, for gRPC clients.

use super::sse::{
    self, ErrorEventScanner, PumpItem, RouterStreamError, StreamEnd, StreamEndReason, StreamLimits,
};
use super::{breaker_outcome, stream_breaker_outcome, BreakerOutcome, Proxy};
use crate::server::error::{router_grpc_status, ApiError};
use crate::server::header_utils::should_forward_request_header;
use crate::workers::Worker;
use axum::http::{HeaderMap, StatusCode};
use bytes::Bytes;
use futures::{Stream, StreamExt, TryStreamExt};
use sglang_grpc_types::sglang::runtime::v1::sglang_service_client::SglangServiceClient;
use sglang_grpc_types::sglang::runtime::v1::{AbortRequest, OpenAiRequest, OpenAiStreamChunk};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;
use tonic::{Code, Status};

/// Engine chunks on their way to a gRPC client.
pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<OpenAiStreamChunk, Status>> + Send>>;

/// An engine's answer to a gRPC call.
pub struct GrpcResponse {
    /// The engine's HTTP status for a non-streaming call; a stream is 200.
    pub status: StatusCode,
    pub chunks: ChunkStream,
}

impl PumpItem for OpenAiStreamChunk {
    fn is_error_event(&self, _: &mut ErrorEventScanner) -> bool {
        sse::is_error_payload(&self.json_chunk)
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
                let abort = client.abort(AbortRequest {
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

fn http_status(chunk: &OpenAiStreamChunk) -> Option<StatusCode> {
    StatusCode::from_u16(u16::try_from(chunk.status_code?).ok()?).ok()
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
    /// Relay `ChatComplete` to `worker`. A streaming call goes through the same
    /// pump as SSE; any other is read to the end within the request timeout.
    /// An unfinished call aborts the engine request `abort_rid`.
    #[allow(clippy::too_many_arguments)]
    pub async fn chat_complete_grpc(
        &self,
        worker: &Worker,
        headers: &HeaderMap,
        body: Bytes,
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
        let request = OpenAiRequest {
            json_body: body.into(),
            trace_headers: trace_headers(headers),
        };
        let failed = |status: &Status| status_breaker_outcome(status.code()).record(breaker);
        let respond = |status, chunks: Vec<_>, mut abort: AbortOnDrop| {
            breaker_outcome(status).record(breaker);
            abort.disarm();
            GrpcResponse {
                status,
                chunks: Box::pin(futures::stream::iter(chunks.into_iter().map(Ok))),
            }
        };
        if !streaming {
            let call = async {
                let mut stream = client.chat_complete(request).await?.into_inner();
                let mut chunks = Vec::new();
                while let Some(chunk) = stream.message().await? {
                    chunks.push(chunk);
                }
                Ok::<_, Status>(chunks)
            };
            let chunks = tokio::time::timeout(self.request_timeout, call)
                .await
                .unwrap_or_else(|_| Err(upstream_timeout()))
                .inspect_err(failed)
                .map_err(ApiError::UpstreamGrpc)?;
            let status = chunks
                .last()
                .and_then(http_status)
                .unwrap_or(StatusCode::OK);
            permit.disarm();
            drop(stream_guards);
            return Ok(respond(status, chunks, abort));
        }
        let call = async {
            let mut stream = client.chat_complete(request).await?.into_inner();
            Ok::<_, Status>((stream.message().await?, stream))
        };
        // The engine answers a request it rejects with one finished chunk carrying
        // an HTTP status, as an HTTP engine answers with a status line.
        let idle = self.stream_idle_timeout.unwrap_or(Duration::MAX);
        let (first, stream) = match tokio::time::timeout(idle, call).await {
            Ok(result) => result.inspect_err(failed).map_err(ApiError::UpstreamGrpc)?,
            Err(_) => {
                breaker.record_failure();
                return Err(ApiError::UpstreamGrpc(upstream_timeout()));
            }
        };
        if let Some(status) = first
            .as_ref()
            .filter(|chunk| chunk.finished)
            .and_then(http_status)
        {
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
        let chunks = sse::pump(
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
            chunks: Box::pin(chunks.map(|chunk| chunk.map_err(into_status))),
        })
    }
}
