// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Header forwarding rules — mirrors SMG semantics.

use axum::http::{header, HeaderMap, HeaderName};

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

/// Copy end-to-end response headers while removing fields that apply only to
/// the worker connection or must be regenerated for the client connection.
pub fn copy_response_headers(headers: &HeaderMap) -> HeaderMap {
    let mut connection_options = Vec::new();
    for value in headers.get_all(header::CONNECTION) {
        connection_options.extend(
            value
                .as_bytes()
                .split(|byte| *byte == b',')
                .filter_map(|name| HeaderName::from_bytes(name.trim_ascii()).ok()),
        );
    }

    let mut copied = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        if should_forward_response_header(name) && !connection_options.contains(name) {
            copied.append(name.clone(), value.clone());
        }
    }
    copied
}

fn should_forward_response_header(name: &HeaderName) -> bool {
    !matches!(
        name.as_str(),
        "connection"
            | "content-length"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderName;

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

    #[test]
    fn response_headers_strip_connection_fields_and_preserve_multiple_values() {
        let mut headers = HeaderMap::new();
        headers.insert("connection", "keep-alive, x-worker-hop".parse().unwrap());
        headers.insert("keep-alive", "timeout=5".parse().unwrap());
        headers.insert("x-worker-hop", "private".parse().unwrap());
        headers.insert("content-length", "123".parse().unwrap());
        headers.insert("retry-after", "7".parse().unwrap());
        headers.append("set-cookie", "a=1".parse().unwrap());
        headers.append("set-cookie", "b=2".parse().unwrap());

        let copied = copy_response_headers(&headers);

        assert!(!copied.contains_key("connection"));
        assert!(!copied.contains_key("keep-alive"));
        assert!(!copied.contains_key("x-worker-hop"));
        assert!(!copied.contains_key("content-length"));
        assert_eq!(copied.get("retry-after").unwrap(), "7");
        assert_eq!(copied.get_all("set-cookie").iter().count(), 2);
    }
}
