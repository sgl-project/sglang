// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Header forwarding whitelist — mirrors SMG semantics.
//!
//! The built-in whitelist covers authentication, tracing, and `x-sgl-*`
//! headers.  Callers can extend the forwarded set at startup via
//! [`ExtraForwardHeaders`], which is populated from the CLI flag
//! `--forward-headers`.

use axum::http::HeaderName;
use std::collections::HashSet;

/// Immutable set of additional header names that should be forwarded to
/// upstream workers.  Built once from `--forward-headers` and shared
/// (read-only) for the lifetime of the process.
#[derive(Debug, Clone, Default)]
pub struct ExtraForwardHeaders {
    names: HashSet<String>,
}

impl ExtraForwardHeaders {
    /// Build from an iterator of header name strings (lowercased
    /// internally).  Invalid HTTP header names are silently skipped.
    pub fn from_iter(iter: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        let names = iter
            .into_iter()
            .filter_map(|s| {
                let lower = s.as_ref().to_ascii_lowercase();
                // Validate that it parses as an HTTP header name.
                HeaderName::try_from(lower.as_str()).ok()?;
                Some(lower)
            })
            .collect();
        Self { names }
    }

    /// True when empty (no CLI flag provided).
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// True if a request header from the inbound client should be forwarded
/// to the upstream worker.
///
/// The built-in whitelist always applies.  When `extra` is non-empty the
/// header is also forwarded if its lowercased name is in the set.
pub fn should_forward_request_header(name: &HeaderName, extra: &ExtraForwardHeaders) -> bool {
    let n = name.as_str();
    matches!(
        n,
        "authorization" | "x-request-id" | "x-correlation-id" | "traceparent" | "tracestate"
    ) || n.starts_with("x-request-id-")
        || n.starts_with("x-sgl-")
        || extra.names.contains(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderName;

    fn empty() -> ExtraForwardHeaders {
        ExtraForwardHeaders::default()
    }

    #[test]
    fn whitelist_basics() {
        let extra = empty();
        // Whitelisted headers
        assert!(should_forward_request_header(
            &HeaderName::from_static("authorization"),
            &extra
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("x-request-id"),
            &extra
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("x-correlation-id"),
            &extra
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("traceparent"),
            &extra
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("tracestate"),
            &extra
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("x-sgl-route-key"),
            &extra
        ));
        assert!(should_forward_request_header(
            &HeaderName::from_static("x-request-id-extra"),
            &extra
        ));

        // Stripped headers
        assert!(!should_forward_request_header(
            &HeaderName::from_static("host"),
            &extra
        ));
        assert!(!should_forward_request_header(
            &HeaderName::from_static("content-length"),
            &extra
        ));
        assert!(!should_forward_request_header(
            &HeaderName::from_static("cookie"),
            &extra
        ));
        assert!(!should_forward_request_header(
            &HeaderName::from_static("connection"),
            &extra
        ));
        assert!(!should_forward_request_header(
            &HeaderName::from_static("transfer-encoding"),
            &extra
        ));
    }

    #[test]
    fn whitelist_prefix_negatives() {
        let extra = empty();
        assert!(
            should_forward_request_header(&HeaderName::from_static("x-request-id"), &extra),
            "x-request-id (exact match) must forward",
        );
        assert!(
            !should_forward_request_header(&HeaderName::from_static("x-request-id2"), &extra),
            "x-request-id2 (no hyphen separator) must not forward",
        );
        assert!(
            !should_forward_request_header(&HeaderName::from_static("x-request-idfoo"), &extra),
            "x-request-idfoo (no hyphen separator) must not forward",
        );
        assert!(
            !should_forward_request_header(&HeaderName::from_static("x-sg-foo"), &extra),
            "x-sg-foo (typo of x-sgl-) must not forward",
        );
        assert!(
            !should_forward_request_header(&HeaderName::from_static("xx-request-id-foo"), &extra),
            "xx-request-id-foo (extra leading char) must not forward",
        );
        assert!(
            !should_forward_request_header(&HeaderName::from_static("xx-sgl-foo"), &extra),
            "xx-sgl-foo (extra leading char) must not forward",
        );
        assert!(
            !should_forward_request_header(&HeaderName::from_static("foo-x-sgl-bar"), &extra),
            "foo-x-sgl-bar (substring, not prefix) must not forward",
        );
    }

    #[test]
    fn extra_headers_forwarded() {
        let extra = ExtraForwardHeaders::from_iter(["x-cloudwalk-info", "x-custom-label"]);
        assert!(
            should_forward_request_header(&HeaderName::from_static("x-cloudwalk-info"), &extra),
            "x-cloudwalk-info must forward when in extra set",
        );
        assert!(
            should_forward_request_header(&HeaderName::from_static("x-custom-label"), &extra),
            "x-custom-label must forward when in extra set",
        );
        // Not in extra set — must not forward
        assert!(
            !should_forward_request_header(&HeaderName::from_static("x-other-header"), &extra),
            "x-other-header must not forward when not in extra set",
        );
    }

    #[test]
    fn extra_headers_empty_preserves_default_behavior() {
        let extra = empty();
        assert!(
            !should_forward_request_header(&HeaderName::from_static("x-cloudwalk-info"), &extra),
            "x-cloudwalk-info must not forward with empty extra set",
        );
    }

    #[test]
    fn extra_headers_case_insensitive() {
        let extra = ExtraForwardHeaders::from_iter(["X-Cloudwalk-Info"]);
        assert!(
            should_forward_request_header(&HeaderName::from_static("x-cloudwalk-info"), &extra),
            "case-insensitive match must work",
        );
    }

    #[test]
    fn extra_headers_invalid_names_skipped() {
        let extra = ExtraForwardHeaders::from_iter(["valid-header", "invalid header with spaces"]);
        assert!(extra.names.contains("valid-header"));
        assert!(!extra.names.contains("invalid header with spaces"));
    }
}
