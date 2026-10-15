// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/responses`: converted to chat and routed by [`chat_completions`];
//! the reply is converted on the outer body.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, Response, StatusCode};
use bytes::Bytes;

use crate::protocol::responses::{
    self, response::chat_to_response, stream::ResponsesStream, EchoContext,
};
use crate::protocol::sse::transduce_body;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::routes::chat::chat_completions;
use crate::server::routes::reply::{is_sse, rebuild, MAX_REPLY_BYTES};

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
    if parts.status.is_success() && streaming && is_sse(&parts) {
        parts.headers.remove(header::CONTENT_LENGTH);
        return Response::from_parts(parts, transduce_body(body, ResponsesStream::new(echo)));
    }

    let status = parts.status.as_u16();
    let bytes = match axum::body::to_bytes(body, MAX_REPLY_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "responses: failed to read upstream reply");
            parts.status = StatusCode::BAD_GATEWAY;
            return rebuild(
                parts,
                responses::error_body(502, "failed to read the upstream reply"),
            );
        }
    };
    if !parts.status.is_success() {
        return rebuild(parts, responses::wrap_error_body(&bytes, status));
    }
    match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(chat) => {
            let body = serde_json::to_vec(&chat_to_response(&chat, &echo)).expect("serialize");
            rebuild(parts, body)
        }
        Err(e) => {
            tracing::warn!(error = %e, "responses: upstream reply is not JSON");
            parts.status = StatusCode::BAD_GATEWAY;
            rebuild(
                parts,
                responses::error_body(502, "upstream reply is not a valid chat completion"),
            )
        }
    }
}
