// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SSE parsing and body-collection helpers for integration tests.

use bytes::Bytes;

/// Parse an SSE stream's `data: …` payloads (one per event).
#[allow(dead_code)]
pub fn parse_sse_data(raw: &[u8]) -> Vec<String> {
    let s = std::str::from_utf8(raw).unwrap_or("");
    s.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|l| l.to_string())
        .collect()
}

/// Collect an axum Body to bytes in tests.
#[allow(dead_code)]
pub async fn collect_body(body: axum::body::Body) -> Bytes {
    use http_body_util::BodyExt;
    body.collect().await.unwrap().to_bytes()
}
