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
    headers.insert(
        HEADER_X_SMG_ERROR_CODE,
        HeaderValue::from_str(&code_str).unwrap(),
    );

    (
        status,
        headers,
        Json(ErrorResponse {
            error: ErrorDetail {
                error_type: status_code_to_str(status),
                code: &code_str,
                message: &message_str,
            },
        }),
    )
        .into_response()
}

fn status_code_to_str(status_code: StatusCode) -> &'static str {
    status_code
        .canonical_reason()
        .unwrap_or("Unknown Status Code")
}

pub fn extract_error_code_from_response<B>(response: &Response<B>) -> &str {
    response
        .headers()
        .get(HEADER_X_SMG_ERROR_CODE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
}
