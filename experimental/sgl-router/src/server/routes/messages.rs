// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/messages` and `/v1/messages/count_tokens`: converted to chat and
//! routed by [`chat_completions`]; the reply is converted on the outer body.
//! Every non-2xx body gets the Anthropic error envelope.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, Response, StatusCode};
use axum::response::IntoResponse;
use bytes::Bytes;

use crate::discovery::ModelId;
use crate::protocol::anthropic::{
    self, response::chat_to_message, stream::MessagesStream, EchoContext,
};
use crate::protocol::transduce_body;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::routes::chat::{chat_completions, MAX_CHAT_BODY_BYTES};

pub(crate) async fn messages(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    let converted = match parse(&body, false) {
        Ok(c) => c,
        Err(e) => return adapt(e.into_response(), None).await,
    };
    let chat_body = Bytes::from(serde_json::to_vec(&converted.chat).expect("serialize chat"));
    let resp = match chat_completions(State(ctx), headers, chat_body).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    };
    adapt(resp, Some((converted.echo, converted.stream))).await
}

/// Exact with the model's chat encoder, approximate without.
pub(crate) async fn count_tokens(
    State(ctx): State<Arc<AppContext>>,
    body: Bytes,
) -> Response<Body> {
    let converted = match parse(&body, true) {
        Ok(c) => c,
        Err(e) => return adapt(e.into_response(), None).await,
    };
    let model = converted.echo.model;
    if model != ctx.config.model.id {
        return adapt(ApiError::ModelNotFound(model).into_response(), None).await;
    }
    match crate::policies::request_tokens_for(&ctx.tokenizers, &ModelId(model), &converted.chat) {
        Some(t) => axum::Json(serde_json::json!({"input_tokens": t.ids.len()})).into_response(),
        None => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [(header::CONTENT_TYPE, "application/json")],
            anthropic::error_body(500, "token counting is unavailable for this model"),
        )
            .into_response(),
    }
}

fn parse(body: &Bytes, count_only: bool) -> Result<anthropic::Converted, ApiError> {
    let req: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::BadRequest(format!("request body is not valid JSON: {e}")))?;
    anthropic::to_chat(req, count_only).map_err(ApiError::BadRequest)
}

/// `echo` is `Some((echo, streaming))`, `None` on the error paths.
async fn adapt(resp: Response<Body>, echo: Option<(EchoContext, bool)>) -> Response<Body> {
    let (mut parts, body) = resp.into_parts();
    let is_sse = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));

    if let Some((echo, true)) = &echo {
        if parts.status.is_success() && is_sse {
            parts.headers.remove(header::CONTENT_LENGTH);
            return Response::from_parts(
                parts,
                transduce_body(body, MessagesStream::new(echo.clone())),
            );
        }
    }

    let status = parts.status.as_u16();
    let bytes = match axum::body::to_bytes(body, MAX_CHAT_BODY_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "messages: failed to read upstream reply");
            parts.status = StatusCode::BAD_GATEWAY;
            return rebuild(
                parts,
                anthropic::error_body(502, "failed to read the upstream reply"),
            );
        }
    };
    if !parts.status.is_success() {
        return rebuild(parts, anthropic::wrap_error_body(&bytes, status));
    }
    let Some((echo, _)) = echo else {
        return Response::from_parts(parts, Body::from(bytes));
    };
    match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(chat) => {
            let body = serde_json::to_vec(&chat_to_message(&chat, &echo)).expect("serialize");
            rebuild(parts, body)
        }
        Err(e) => {
            tracing::warn!(error = %e, "messages: upstream reply is not JSON");
            parts.status = StatusCode::BAD_GATEWAY;
            rebuild(
                parts,
                anthropic::error_body(502, "upstream reply is not a valid chat completion"),
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
