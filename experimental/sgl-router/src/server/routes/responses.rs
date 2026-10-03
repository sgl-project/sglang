// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/responses`: converted to chat and routed by [`chat_completions`];
//! the reply is converted on the outer body.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, Response};
use bytes::Bytes;

use crate::protocol::responses::{
    self, response::chat_to_response, stream::ResponsesStream, EchoContext,
};
use crate::protocol::transduce_body;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::routes::chat::chat_completions;

pub(super) const MAX_REPLY_BYTES: usize = crate::server::routes::chat::MAX_CHAT_BODY_BYTES;

pub(crate) async fn responses(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let req: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::BadRequest(format!("request body is not valid JSON: {e}")))?;
    let converted = responses::to_chat(req).map_err(ApiError::BadRequest)?;
    let chat_body = Bytes::from(
        serde_json::to_vec(&converted.chat)
            .map_err(|e| ApiError::Internal(anyhow::anyhow!("serialize chat request: {e}")))?,
    );
    let resp = chat_completions(State(ctx), headers, chat_body).await?;
    Ok(adapt(resp, converted.echo, converted.stream).await)
}

/// Keeps status, headers and extensions (access log); replaces the body.
async fn adapt(resp: Response<Body>, echo: EchoContext, streaming: bool) -> Response<Body> {
    let (mut parts, body) = resp.into_parts();
    let is_sse = parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));

    if parts.status.is_success() && streaming && is_sse {
        parts.headers.remove(header::CONTENT_LENGTH);
        return Response::from_parts(parts, transduce_body(body, ResponsesStream::new(echo)));
    }

    let bytes = match axum::body::to_bytes(body, MAX_REPLY_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "responses: failed to read upstream reply");
            return rebuild(parts, error_json(502, "failed to read the upstream reply"));
        }
    };

    if !parts.status.is_success() {
        return match responses::wrap_error_body(&bytes, parts.status.as_u16()) {
            Some(wrapped) => rebuild(parts, wrapped),
            None => Response::from_parts(parts, Body::from(bytes)),
        };
    }

    match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(chat) => {
            let converted = chat_to_response(&chat, &echo);
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

pub(super) fn rebuild(mut parts: axum::http::response::Parts, body: Vec<u8>) -> Response<Body> {
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
