// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response};
use bytes::Bytes;
use serde_json::Value;
use std::sync::Arc;

/// POST /v1/chat/completions — forward to the worker. M1 routes to a
/// single hardcoded worker (no policy). If the request opts into
/// streaming (`stream: true`), we pipe SSE bytes back; otherwise
/// buffer.
pub async fn chat_completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let streaming = parse_streaming(&body)?;
    if streaming {
        ctx.proxy
            .forward_streaming("/v1/chat/completions", &headers, body)
            .await
    } else {
        ctx.proxy
            .forward_json("/v1/chat/completions", &headers, body)
            .await
    }
}

fn parse_streaming(body: &Bytes) -> Result<bool, ApiError> {
    let v: Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::BadRequest(format!("invalid JSON body: {e}")))?;
    Ok(v.get("stream").and_then(|x| x.as_bool()).unwrap_or(false))
}
