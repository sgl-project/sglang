// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Authentication helpers for inbound router-admin endpoints.

use axum::http::{header::AUTHORIZATION, HeaderMap};
use subtle::ConstantTimeEq;

/// Returns whether a request is authorized for router-admin endpoints.
///
/// `expected_api_key = None` preserves the legacy behavior: admin endpoints
/// are open unless an operator explicitly configures a router admin key.
pub fn is_admin_authorized(headers: &HeaderMap, expected_api_key: Option<&str>) -> bool {
    let Some(expected_api_key) = expected_api_key else {
        return true;
    };

    let Some(header) = headers.get(AUTHORIZATION) else {
        return false;
    };
    let Ok(header) = header.to_str() else {
        return false;
    };
    let Some((scheme, token)) = header.split_once(' ') else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("bearer") {
        return false;
    }

    token.as_bytes().ct_eq(expected_api_key.as_bytes()).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers_with_authorization(value: HeaderValue) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, value);
        headers
    }

    fn headers_with_authorization_str(value: &str) -> HeaderMap {
        headers_with_authorization(HeaderValue::from_str(value).unwrap())
    }

    #[test]
    fn no_configured_key_allows_missing_header() {
        assert!(is_admin_authorized(&HeaderMap::new(), None));
    }

    #[test]
    fn configured_key_rejects_missing_header() {
        assert!(!is_admin_authorized(&HeaderMap::new(), Some("secret")));
    }

    #[test]
    fn configured_key_accepts_exact_bearer_token() {
        let headers = headers_with_authorization_str("Bearer secret");
        assert!(is_admin_authorized(&headers, Some("secret")));
    }

    #[test]
    fn configured_key_accepts_case_insensitive_bearer_scheme() {
        let headers = headers_with_authorization_str("bEaReR secret");
        assert!(is_admin_authorized(&headers, Some("secret")));
    }

    #[test]
    fn configured_key_rejects_wrong_scheme() {
        let headers = headers_with_authorization_str("Basic secret");
        assert!(!is_admin_authorized(&headers, Some("secret")));
    }

    #[test]
    fn configured_key_rejects_malformed_authorization() {
        let headers = headers_with_authorization_str("secret");
        assert!(!is_admin_authorized(&headers, Some("secret")));
    }

    #[test]
    fn configured_key_rejects_wrong_token() {
        let headers = headers_with_authorization_str("Bearer wrong");
        assert!(!is_admin_authorized(&headers, Some("secret")));
    }

    #[test]
    fn configured_key_rejects_token_with_extra_spaces() {
        let leading = headers_with_authorization_str("Bearer  secret");
        let trailing = headers_with_authorization_str("Bearer secret ");
        assert!(!is_admin_authorized(&leading, Some("secret")));
        assert!(!is_admin_authorized(&trailing, Some("secret")));
    }

    #[test]
    fn configured_key_rejects_non_visible_ascii_header() {
        let headers = headers_with_authorization(HeaderValue::from_bytes(b"Bearer \xff").unwrap());
        assert!(!is_admin_authorized(&headers, Some("secret")));
    }
}
