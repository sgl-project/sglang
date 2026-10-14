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

#[cfg(test)]
mod tests {
    use super::*;

    /// Ported from SMG tests/api/streaming_tests.rs::test_sse_format_parsing.
    /// Verifies that parse_sse_data:
    ///   1. Extracts standard `data: …` lines.
    ///   2. Silently ignores SSE `event: …` type fields (not data lines).
    ///   3. Silently ignores SSE `: …` comment lines.
    ///   4. Correctly parses `[DONE]` sentinel.
    ///
    /// These edge-cases matter because SGLang workers may emit `event: message`
    /// fields in their SSE frames. A parser that accidentally leaks those into
    /// the payload list would cause clients to fail on JSON-parse.
    #[test]
    fn parse_sse_data_extracts_data_lines_only() {
        // Basic: three data lines including the [DONE] sentinel.
        let basic =
            b"data: {\"text\":\"Hello\"}\n\ndata: {\"text\":\" world\"}\n\ndata: [DONE]\n\n";
        let events = parse_sse_data(basic);
        assert_eq!(events.len(), 3, "expected 3 data events, got: {events:?}");
        assert_eq!(events[0], "{\"text\":\"Hello\"}");
        assert_eq!(events[1], "{\"text\":\" world\"}");
        assert_eq!(events[2], "[DONE]");

        // Mixed: event: type field + comment line — neither must appear in output.
        let mixed = b"event: message\ndata: {\"test\":true}\n\n: comment line\ndata: [DONE]\n\n";
        let events = parse_sse_data(mixed);
        assert_eq!(
            events.len(),
            2,
            "event: and : comment lines must be ignored; got: {events:?}"
        );
        assert_eq!(events[0], "{\"test\":true}");
        assert_eq!(events[1], "[DONE]");
    }
}
