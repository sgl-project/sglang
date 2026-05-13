use axum::{
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

#[derive(Serialize)]
struct ErrorResponse<'a> {
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'a str,
    message: &'a str,
}

pub const HEADER_X_SMG_ERROR_CODE: &str = "X-SMG-Error-Code";

pub fn internal_error(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::INTERNAL_SERVER_ERROR, code, message)
}

pub fn bad_request(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::BAD_REQUEST, code, message)
}

pub fn not_found(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_FOUND, code, message)
}

pub fn service_unavailable(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::SERVICE_UNAVAILABLE, code, message)
}

pub fn failed_dependency(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::FAILED_DEPENDENCY, code, message)
}

pub fn not_implemented(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_IMPLEMENTED, code, message)
}

pub fn bad_gateway(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::BAD_GATEWAY, code, message)
}

pub fn method_not_allowed(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::METHOD_NOT_ALLOWED, code, message)
}

pub fn create_error(
    status: StatusCode,
    code: impl Into<String>,
    message: impl Into<String>,
) -> Response {
    let code_str = code.into();
    let message_str = message.into();

    let mut headers = HeaderMap::with_capacity(1);
    // `code_str` is developer-controlled today, but unwrap would panic if a
    // future caller smuggled in a byte that's invalid inside an HTTP header
    // value (CTL chars like `\n`, or non-visible bytes). Fall back to a
    // static sentinel so the response status/body still propagate.
    headers.insert(
        HEADER_X_SMG_ERROR_CODE,
        HeaderValue::from_str(&code_str)
            .unwrap_or_else(|_| HeaderValue::from_static("unknown_error")),
    );

    (
        status,
        headers,
        Json(ErrorResponse {
            error: ErrorDetail {
                error_type: error_type_for_status(status),
                code: &code_str,
                message: &message_str,
            },
        }),
    )
        .into_response()
}

/// OpenAI-compat error class string. SDK clients classify on this
/// field — `RateLimitError`, `AuthenticationError`, etc. are
/// dispatched off the `error.type` value, so a 429 must come back as
/// `rate_limit_error` (not the catch-all `invalid_request_error`)
/// for client retry/backoff logic to fire.
fn error_type_for_status(status: StatusCode) -> &'static str {
    match status {
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        s if s.is_client_error() => "invalid_request_error",
        s if s.is_server_error() => "server_error",
        _ => "api_error",
    }
}

pub fn extract_error_code_from_response<B>(response: &Response<B>) -> &str {
    response
        .headers()
        .get(HEADER_X_SMG_ERROR_CODE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `create_error` must not panic when a caller smuggles a byte that
    /// is invalid in an HTTP header value (here `\n` — a CTL char, not
    /// strictly non-ASCII). Without the `unwrap_or_else` fallback the
    /// previous `unwrap()` would have taken down the request handler.
    #[test]
    fn invalid_header_byte_code_falls_back_to_sentinel_header() {
        let response = create_error(StatusCode::BAD_REQUEST, "bad\ncode", "msg");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let header = response
            .headers()
            .get(HEADER_X_SMG_ERROR_CODE)
            .expect("error code header must be present");
        assert_eq!(header.to_str().unwrap(), "unknown_error");
    }

    #[test]
    fn ascii_code_round_trips_in_header() {
        let response = create_error(StatusCode::BAD_REQUEST, "json_parse_error", "msg");
        let header = response
            .headers()
            .get(HEADER_X_SMG_ERROR_CODE)
            .expect("error code header must be present");
        assert_eq!(header.to_str().unwrap(), "json_parse_error");
    }
}
