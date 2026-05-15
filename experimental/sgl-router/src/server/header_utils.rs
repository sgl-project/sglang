// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Header forwarding whitelist — mirrors SMG semantics.

use axum::http::HeaderName;

/// True if a request header from the inbound client should be forwarded
/// to the upstream worker. Mirrors SMG's whitelist semantics.
pub fn should_forward_request_header(name: &HeaderName) -> bool {
    let n = name.as_str();
    matches!(
        n,
        "authorization"
            | "x-request-id"
            | "x-correlation-id"
            | "traceparent"
            | "tracestate"
    ) || n.starts_with("x-request-id-")
        || n.starts_with("x-sgl-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderName;

    #[test]
    fn whitelist_basics() {
        // Whitelisted headers
        assert!(should_forward_request_header(
            &HeaderName::from_static("authorization")
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("x-request-id")
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("x-correlation-id")
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("traceparent")
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("tracestate")
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("x-sgl-route-key")
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("x-request-id-extra")
        ));

        // Stripped headers
        assert!(!should_forward_request_header(
            &HeaderName::from_static("host")
        ));
        assert!(!should_forward_request_header(
            &HeaderName::from_static("content-length")
        ));
        assert!(!should_forward_request_header(
            &HeaderName::from_static("cookie")
        ));
        assert!(!should_forward_request_header(
            &HeaderName::from_static("connection")
        ));
        assert!(!should_forward_request_header(
            &HeaderName::from_static("transfer-encoding")
        ));
    }
}
