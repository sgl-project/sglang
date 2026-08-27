//! OpenAI-compatible generation endpoints.
//!
//! The HTTP adapter stays deliberately thin: Dynamo owns the standard OpenAI
//! request and response primitives. Native [`ChunkEvent`] values remain the one
//! backend output type for both unary and streaming responses.

use axum::{Router, http::StatusCode, response::Response};
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::mpsc;

mod chat;
mod completions;
mod models;

use super::app::AppState;
use super::frame::OutputAccumulator;
use super::submit::submit;
use crate::frontend::AbortGuard;
use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, RequestKind};
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::utils::response::error_response;

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .merge(models::routes())
        .merge(completions::routes())
        .merge(chat::routes())
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn unix_seconds_u32() -> u32 {
    u32::try_from(unix_seconds()).unwrap_or(u32::MAX)
}

/// The OpenAI error payload.
pub(super) fn error_payload(code: StatusCode, message: impl Into<String>) -> serde_json::Value {
    let message = message.into();
    let error_type = if code == StatusCode::UNAUTHORIZED {
        "AuthenticationError"
    } else if code.is_server_error() {
        "InternalServerError"
    } else {
        "BadRequestError"
    };
    serde_json::json!({
        "error": {
            "object": "error",
            "message": message,
            "type": error_type,
            "param": null,
            "code": code.as_u16(),
        }
    })
}

/// Form an OpenAI error response: unary → `code` plus the JSON `body`,
/// streaming → 200 with one SSE error frame + `[DONE]`.
pub(super) fn openai_error(code: StatusCode, message: impl Into<String>, stream: bool) -> Response {
    error_response(code, error_payload(code, message), stream)
}

/// Drain one submitted request to its terminal output: fold frames, disarm
/// `guard` on a natural terminal, and map errors / validation aborts /
/// truncation to `(status, message)` for the OpenAI error shape.
async fn collect_output(
    mut rx: mpsc::Receiver<ResponseItem>,
    guard: &mut AbortGuard,
    rid: &Rid,
) -> Result<ChunkEvent, (StatusCode, String)> {
    let mut accumulator = OutputAccumulator::default();
    let output = loop {
        match rx.recv().await {
            Some(ResponseItem::Frame(output)) => accumulator.fold(&output),
            Some(ResponseItem::Done(output)) => {
                accumulator.fold(&output);
                break accumulator.into_output();
            }
            Some(ResponseItem::Error(error)) => {
                guard.disarm(rid);
                let status = StatusCode::from_u16(error.http_status())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return Err((status, error.to_string()));
            }
            Some(ResponseItem::Control(_)) | Some(ResponseItem::Data(_)) => {}
            None => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "response truncated before completion".into(),
                ));
            }
        }
    };
    guard.disarm(rid);
    if let Some((code, message)) = output
        .finish_reason
        .as_ref()
        .and_then(|reason| reason.abort_status())
    {
        return Err((
            StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            message.to_owned(),
        ));
    }
    Ok(output)
}

async fn submit_generation(
    state: &AppState,
    request: GenerateRequest,
    stream: bool,
    guard: &mut AbortGuard,
) -> Result<mpsc::Receiver<ResponseItem>, Response> {
    match submit(state, RequestKind::Generate(Box::new(request)), stream).await {
        Ok((rid, rx)) => {
            guard.arm(rid);
            Ok(rx)
        }
        // Same `error_response` rule: a committed stream gets 200 plus an
        // SSE error frame + `[DONE]`, not a unary 503 — but with the OpenAI
        // error shape, since this is the OpenAI frontend.
        Err(_) => Err(openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable",
            stream,
        )),
    }
}

fn indexed_decode_stream(
    index: usize,
    rx: mpsc::Receiver<ResponseItem>,
) -> futures::stream::BoxStream<'static, (usize, Option<ResponseItem>)> {
    futures::stream::unfold((rx, false), move |(mut rx, finished)| async move {
        if finished {
            return None;
        }
        match rx.recv().await {
            Some(item) => {
                let finished = matches!(item, ResponseItem::Done(_) | ResponseItem::Error(_));
                Some(((index, Some(item)), (rx, finished)))
            }
            None => Some(((index, None), (rx, true))),
        }
    })
    .boxed()
}

#[cfg(test)]
mod test_utils;
