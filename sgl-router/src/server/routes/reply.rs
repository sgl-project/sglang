// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared by the routes that convert a chat reply into another protocol.

use axum::body::Body;
use axum::http::response::Parts;
use axum::http::{header, HeaderValue, Response};

use crate::discovery::ModelId;
use crate::server::app_context::AppContext;
use crate::server::routes::chat::MAX_CHAT_BODY_BYTES;

/// Cap on a buffered non-streaming upstream reply. Pinned to the compiled-in
/// request default, not the configured request cap: a reply grows with the
/// generated output, not with the prompt, so raising the request cap for large
/// prompts should not raise per-reply memory, and lowering it should not start
/// failing large replies.
pub(super) const MAX_REPLY_BYTES: usize = MAX_CHAT_BODY_BYTES;

/// Whether every worker of the served model renders a system message after
/// the first; a worker that does not report it counts as no.
pub(super) fn model_supports_inline_system(ctx: &AppContext) -> bool {
    let workers = ctx
        .registry
        .workers_for(&ModelId(ctx.config.model.id.clone()));
    !workers.is_empty() && workers.iter().all(|w| w.supports_inline_system())
}

pub(super) fn is_sse(parts: &Parts) -> bool {
    parts
        .headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"))
}

/// A JSON body on the upstream reply's status, headers and extensions.
pub(super) fn rebuild(mut parts: Parts, body: Vec<u8>) -> Response<Body> {
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    Response::from_parts(parts, Body::from(body))
}
