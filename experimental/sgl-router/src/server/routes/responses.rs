// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/responses` — the OpenAI Responses API, served by converting to
//! a chat request and running it through [`chat_completions_inner`], so a
//! Responses request is routed exactly like a chat one (PD bootstrap, ingress
//! tokenize + `input_ids`, cache-aware routing, admission, abort, retry).
//!
//! The conversion back wraps the OUTER response body: the SSE pump's TTFT /
//! ITL hooks, in-band error scanner, extend-tee capture and abort-on-drop all
//! still see the engine's chat bytes. Dropping this body (client disconnect)
//! drops the inner one, so the engine abort fires as it does for chat.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, Response};
use axum::Extension;
use bytes::Bytes;
use futures::StreamExt;

use crate::protocol::responses::{
    self, response::chat_to_response, stream::ResponsesStream, EchoContext,
};
use crate::server::app::RequestPhaseCell;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::routes::chat::chat_completions_inner;

/// Cap when buffering an engine reply for conversion. Matches the request
/// cap: a chat completion body is never larger than the prompt it answers
/// plus the output budget, and the engine's own replies are far below this.
const MAX_REPLY_BYTES: usize = crate::server::routes::chat::MAX_CHAT_BODY_BYTES;

pub(crate) async fn responses(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    phase: Option<Extension<Arc<RequestPhaseCell>>>,
    // Body-consuming extractor: MUST stay last (see `chat_completions`).
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let req: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::BadRequest(format!("request body is not valid JSON: {e}")))?;
    let converted = responses::to_chat(req).map_err(ApiError::BadRequest)?;
    let chat_body = Bytes::from(
        serde_json::to_vec(&converted.chat)
            .map_err(|e| ApiError::Internal(anyhow::anyhow!("serialize chat request: {e}")))?,
    );
    let resp = chat_completions_inner(ctx, headers, phase.map(|Extension(p)| p), chat_body).await?;
    Ok(adapt(resp, converted.echo, converted.stream).await)
}

/// Convert the chat handler's response. `parts` (status, headers and the
/// extensions the access-log middleware reads) are kept; only the body and
/// its framing headers change.
async fn adapt(resp: Response<Body>, echo: EchoContext, streaming: bool) -> Response<Body> {
    let (mut parts, body) = resp.into_parts();
    let is_sse = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));

    if parts.status.is_success() && streaming && is_sse {
        let mut conv = ResponsesStream::new(echo);
        let mut inner = body.into_data_stream();
        // Set once `inner` has yielded `None` (or an error) — it must not be
        // polled again after that.
        let mut inner_done = false;
        let out = futures::stream::poll_fn(move |cx| loop {
            use std::task::Poll;
            if inner_done {
                return Poll::Ready(None);
            }
            if conv.is_terminal() {
                // Drain (and discard) the rest so the pump sees a normal end
                // rather than a client disconnect.
                return match inner.poll_next_unpin(cx) {
                    Poll::Ready(Some(_)) => continue,
                    Poll::Ready(None) => {
                        inner_done = true;
                        Poll::Ready(None)
                    }
                    Poll::Pending => Poll::Pending,
                };
            }
            let bytes = match inner.poll_next_unpin(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => conv.feed(&chunk),
                Poll::Ready(Some(Err(e))) => {
                    tracing::warn!(error = %e, "responses stream: upstream body error");
                    inner_done = true;
                    let out = conv.fail("upstream stream interrupted");
                    return Poll::Ready(Some(Ok::<_, std::io::Error>(Bytes::from(out))));
                }
                Poll::Ready(None) => {
                    inner_done = true;
                    let out = conv.finish();
                    if out.is_empty() {
                        return Poll::Ready(None);
                    }
                    return Poll::Ready(Some(Ok(Bytes::from(out))));
                }
            };
            if !bytes.is_empty() {
                return Poll::Ready(Some(Ok(Bytes::from(bytes))));
            }
        });
        parts.headers.remove(header::CONTENT_LENGTH);
        return Response::from_parts(parts, Body::from_stream(out));
    }

    let bytes = match axum::body::to_bytes(body, MAX_REPLY_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "responses: failed to read upstream reply");
            return rebuild(parts, error_json(502, "failed to read the upstream reply"));
        }
    };

    if !parts.status.is_success() {
        // Engine / router errors: keep the status, normalize the envelope.
        return match responses::wrap_error_body(&bytes, parts.status.as_u16()) {
            Some(wrapped) => rebuild(parts, wrapped),
            None => Response::from_parts(parts, Body::from(bytes)),
        };
    }

    match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(chat) => {
            let converted = chat_to_response(&chat, &echo);
            // Serializing a `Value` cannot fail.
            let body = serde_json::to_vec(&converted).expect("serialize response");
            rebuild(parts, body)
        }
        Err(e) => {
            tracing::warn!(error = %e, "responses: upstream reply is not JSON");
            let mut parts = parts;
            parts.status = axum::http::StatusCode::BAD_GATEWAY;
            rebuild(
                parts,
                error_json(502, "upstream reply is not a valid chat completion"),
            )
        }
    }
}

fn rebuild(mut parts: axum::http::response::Parts, body: Vec<u8>) -> Response<Body> {
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    Response::from_parts(parts, Body::from(body))
}

fn error_json(status: u16, message: &str) -> Vec<u8> {
    let typ = if (400..500).contains(&status) {
        "invalid_request_error"
    } else {
        "server_error"
    };
    serde_json::to_vec(&serde_json::json!({
        "error": {"message": message, "type": typ, "param": null, "code": status}
    }))
    .expect("serialize error")
}
