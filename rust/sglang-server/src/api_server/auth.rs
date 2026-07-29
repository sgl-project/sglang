//! API-key boundary matching the Python HTTP server's normal-endpoint policy.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::Request,
    http::{HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::runtime::ServerArgs;

/// Install authentication only when an API key is configured.
pub(super) fn apply(app: Router, server_args: &ServerArgs) -> Router {
    match server_args.api_key.as_deref() {
        Some(api_key) => {
            let api_key: Arc<str> = Arc::from(api_key);
            app.layer(axum::middleware::from_fn(move |request, next| {
                authenticate(request, next, Arc::clone(&api_key))
            }))
        }
        None => app,
    }
}

async fn authenticate(request: Request, next: Next, api_key: Arc<str>) -> Response {
    let authorization = request.headers().get(axum::http::header::AUTHORIZATION);
    if is_authorized(
        request.method(),
        request.uri().path(),
        authorization,
        &api_key,
    ) {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "Unauthorized"})),
        )
            .into_response()
    }
}

fn is_authorized(
    method: &Method,
    path: &str,
    authorization: Option<&HeaderValue>,
    api_key: &str,
) -> bool {
    if method == Method::OPTIONS
        || path.starts_with("/health")
        || path == "/readiness"
        || path.starts_with("/metrics")
    {
        return true;
    }

    let Some(header) = authorization.and_then(|value| value.to_str().ok()) else {
        return false;
    };
    let Some((scheme, token)) = header.split_once(' ') else {
        return false;
    };
    scheme.eq_ignore_ascii_case("bearer") && constant_time_eq(token.as_bytes(), api_key.as_bytes())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut different = left.len() ^ right.len();
    let count = left.len().max(right.len());
    for index in 0..count {
        let left_byte = left.get(index).copied().unwrap_or(0);
        let right_byte = right.get(index).copied().unwrap_or(0);
        different |= usize::from(left_byte ^ right_byte);
    }
    different == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(value: &'static str) -> HeaderValue {
        HeaderValue::from_static(value)
    }

    #[test]
    fn business_routes_require_matching_bearer_token() {
        assert!(!is_authorized(&Method::POST, "/generate", None, "secret"));
        assert!(!is_authorized(
            &Method::POST,
            "/generate",
            Some(&header("Bearer wrong")),
            "secret"
        ));
        assert!(!is_authorized(
            &Method::POST,
            "/generate",
            Some(&header("Basic secret")),
            "secret"
        ));
        assert!(is_authorized(
            &Method::POST,
            "/generate",
            Some(&header("bEaReR secret")),
            "secret"
        ));
    }

    #[test]
    fn operations_routes_and_options_bypass_authentication() {
        for path in [
            "/health",
            "/health_generate",
            "/readiness",
            "/metrics",
            "/metrics/prometheus",
        ] {
            assert!(is_authorized(&Method::GET, path, None, "secret"), "{path}");
        }
        assert!(is_authorized(&Method::OPTIONS, "/generate", None, "secret"));
    }

    #[test]
    fn token_comparison_is_exact() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret-extra"));
    }
}
