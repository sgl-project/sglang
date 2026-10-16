// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared by the routes that convert a chat reply into another protocol.

use axum::body::Body;
use axum::http::response::Parts;
use axum::http::{header, HeaderValue, Response};

use crate::server::routes::chat::MAX_CHAT_BODY_BYTES;

/// Upstream replies are buffered up to the same cap as requests.
pub(super) const MAX_REPLY_BYTES: usize = MAX_CHAT_BODY_BYTES;

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
