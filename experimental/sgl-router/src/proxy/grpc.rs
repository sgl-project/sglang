// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Forwarding to an engine's native gRPC `SglangService`, for gRPC clients.

use super::sse::{self, ErrorEventScanner, PumpItem, StreamEnd, StreamLimits};
use super::{breaker_outcome, stream_breaker_outcome, BreakerOutcome, Proxy};
use crate::server::error::ApiError;
use crate::server::header_utils::should_forward_request_header;
use crate::workers::Worker;
use axum::http::{HeaderMap, StatusCode};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use sglang_grpc_types::sglang::runtime::v1::{OpenAiRequest, OpenAiStreamChunk};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
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
    /// Relay `ChatComplete` to `worker`. A streaming call goes through the same
    /// pump as SSE; any other is read to the end within the request timeout.
    #[allow(clippy::too_many_arguments)]
    pub async fn chat_complete_grpc(
        &self,
        worker: &Worker,
        headers: &HeaderMap,
        body: Bytes,
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
        let request = OpenAiRequest {
            json_body: body.into(),
            trace_headers: trace_headers(headers),
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
                .unwrap_or_else(|_| Err(Status::deadline_exceeded("upstream request timed out")))
                .inspect_err(|status| status_breaker_outcome(status).record(breaker))
                .map_err(ApiError::UpstreamGrpc)?;
            let status = chunks
                .last()
                .and_then(|chunk| chunk.status_code)
                .and_then(|code| StatusCode::from_u16(u16::try_from(code).ok()?).ok())
                .unwrap_or(StatusCode::OK);
            breaker_outcome(status).record(breaker);
            permit.disarm();
            drop(stream_guards);
            return Ok(GrpcResponse {
                status,
                chunks: Box::pin(futures::stream::iter(chunks.into_iter().map(Ok))),
            });
        }
        let stream = client
            .chat_complete(request)
            .await
            .inspect_err(|status| status_breaker_outcome(status).record(breaker))
            .map_err(ApiError::UpstreamGrpc)?
            .into_inner();
        let breaker = Arc::clone(breaker);
        let on_complete: Box<dyn FnOnce(StreamEnd) + Send + 'static> = Box::new(move |end| {
            stream_breaker_outcome(end).record(&breaker);
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
