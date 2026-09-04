// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Header forwarding whitelist — mirrors SMG semantics.

use axum::http::HeaderName;

/// `Server-Timing` response header name, shared by every site that appends a
/// metric to it — e.g. `router.ttfb` on 2xx streaming responses (`chat.rs`),
/// `router.stage` on router-generated errors (`error.rs`). One definition so
/// the header name itself can't drift between call sites.
pub const SERVER_TIMING: HeaderName = HeaderName::from_static("server-timing");

/// True if a request header from the inbound client should be forwarded
/// to the upstream worker. Mirrors SMG's whitelist semantics.
pub fn should_forward_request_header(name: &HeaderName) -> bool {
    let n = name.as_str();
    matches!(
        n,
        "authorization" | "x-request-id" | "x-correlation-id" | "traceparent" | "tracestate"
    ) || n.starts_with("x-request-id-")
        || n.starts_with("x-sgl-")
}

/// True if an upstream worker's RESPONSE header should be forwarded to the
/// client. The proxy builds downstream responses from scratch (synthesizing
/// content-type), so anything not whitelisted here is silently dropped.
/// Needed for the Kimi accounting headers (`X-Msh-Usage-*`, stream spec
/// P0.18), which the engine sets so clients can account prompt tokens even
/// when a stream drops before the final usage frame; `server-timing` lets the
/// engine's own metrics survive next to the router's appended ones.
/// Hop-by-hop headers and anything the proxy synthesizes itself
/// (content-type/length, transfer-encoding, connection) stay excluded.
pub fn should_forward_response_header(name: &HeaderName) -> bool {
    let n = name.as_str();
    n.starts_with("x-msh-") || n == "server-timing" || n == "x-request-id"
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderName;

    #[test]
    fn response_whitelist_basics() {
        // Forwarded: Kimi accounting headers, engine timing, request id.
        for n in [
            "x-msh-usage-prompt-tokens",
            "x-msh-usage-cached-tokens",
            "server-timing",
            "x-request-id",
        ] {
            assert!(should_forward_response_header(&HeaderName::from_static(n)), "{n}");
        }
        // Dropped: hop-by-hop and proxy-synthesized headers.
        for n in [
            "content-type",
            "content-length",
            "transfer-encoding",
            "connection",
            "x-sgl-decode-url",
            "set-cookie",
        ] {
            assert!(!should_forward_response_header(&HeaderName::from_static(n)), "{n}");
        }
    }

    #[test]
    fn whitelist_basics() {
        // Whitelisted headers
        assert!(should_forward_request_header(&HeaderName::from_static(
            "authorization"
        )));
        assert!(should_forward_request_header(&HeaderName::from_static(
            "x-request-id"
        )));
        assert!(should_forward_request_header(&HeaderName::from_static(
            "x-correlation-id"
        )));
        assert!(should_forward_request_header(&HeaderName::from_static(
            "traceparent"
        )));
        assert!(should_forward_request_header(&HeaderName::from_static(
            "tracestate"
        )));
        assert!(should_forward_request_header(&HeaderName::from_static(
            "x-sgl-route-key"
        )));
        assert!(should_forward_request_header(&HeaderName::from_static(
            "x-request-id-extra"
        )));

        // Stripped headers
        assert!(!should_forward_request_header(&HeaderName::from_static(
            "host"
        )));
        assert!(!should_forward_request_header(&HeaderName::from_static(
            "content-length"
        )));
        assert!(!should_forward_request_header(&HeaderName::from_static(
            "cookie"
        )));
        assert!(!should_forward_request_header(&HeaderName::from_static(
            "connection"
        )));
        assert!(!should_forward_request_header(&HeaderName::from_static(
            "transfer-encoding"
        )));
    }

    /// Prefix-match negatives: names that LOOK similar to `x-request-id-*`
    /// or `x-sgl-*` but must NOT be forwarded. Guards against a future
    /// regression that loosens the rule (e.g., a `contains` instead of
    /// `starts_with`, or a missing hyphen anchor).
    #[test]
    fn whitelist_prefix_negatives() {
        // `x-request-id` itself is an exact match and MUST forward —
        // pin this so a future "tighten prefix to require trailing hyphen"
        // refactor doesn't silently drop the canonical name.
        assert!(
            should_forward_request_header(&HeaderName::from_static("x-request-id")),
            "x-request-id (exact match) must forward",
        );

        // No trailing hyphen between `id` and the suffix: not a child of
        // `x-request-id-*`, must NOT forward.
        assert!(
            !should_forward_request_header(&HeaderName::from_static("x-request-id2")),
            "x-request-id2 (no hyphen separator) must not forward",
        );
        assert!(
            !should_forward_request_header(&HeaderName::from_static("x-request-idfoo")),
            "x-request-idfoo (no hyphen separator) must not forward",
        );

        // Typo of the `x-sgl-` prefix (missing 'l'): must NOT forward.
        assert!(
            !should_forward_request_header(&HeaderName::from_static("x-sg-foo")),
            "x-sg-foo (typo of x-sgl-) must not forward",
        );

        // Extra leading character: `xx-request-id-foo` does not start with
        // `x-request-id-`, must NOT forward.
        assert!(
            !should_forward_request_header(&HeaderName::from_static("xx-request-id-foo")),
            "xx-request-id-foo (extra leading char) must not forward",
        );
        // Same shape for the x-sgl- family.
        assert!(
            !should_forward_request_header(&HeaderName::from_static("xx-sgl-foo")),
            "xx-sgl-foo (extra leading char) must not forward",
        );

        // Substring-but-not-prefix: must NOT forward (guards against a
        // `contains`-based regression).
        assert!(
            !should_forward_request_header(&HeaderName::from_static("foo-x-sgl-bar")),
            "foo-x-sgl-bar (substring, not prefix) must not forward",
        );
    }
}
