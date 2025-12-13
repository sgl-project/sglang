use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

pub fn internal_error(message: impl Into<String>) -> Response {
    create_error(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
}

pub fn bad_request(message: impl Into<String>) -> Response {
    create_error(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

pub fn not_found(message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_FOUND, "invalid_request_error", message)
}

pub fn service_unavailable(message: impl Into<String>) -> Response {
    create_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable",
        message,
    )
}

pub fn failed_dependency(message: impl Into<String>) -> Response {
    create_error(
        StatusCode::FAILED_DEPENDENCY,
        "external_connector_error",
        message,
    )
}

pub fn not_implemented(message: impl Into<String>) -> Response {
    create_error(
        StatusCode::NOT_IMPLEMENTED,
        "not_implemented_error",
        message,
    )
}

fn create_error(status_code: StatusCode, error_type: &str, message: impl Into<String>) -> Response {
    let msg = message.into();
    (
        status_code,
        Json(json!({
            "error": {
                "message": msg,
                "type": error_type,
                "code": status_code.as_u16()
            }
        })),
    )
        .into_response()
}

fn status_code_to_str(status_code: StatusCode) -> &'static str {
    match status_code {
        StatusCode::CONTINUE => "continue",
        StatusCode::SWITCHING_PROTOCOLS => "switching_protocols",
        StatusCode::PROCESSING => "processing",
        StatusCode::OK => "ok",
        StatusCode::CREATED => "created",
        StatusCode::ACCEPTED => "accepted",
        StatusCode::NON_AUTHORITATIVE_INFORMATION => "non_authoritative_information",
        StatusCode::NO_CONTENT => "no_content",
        StatusCode::RESET_CONTENT => "reset_content",
        StatusCode::PARTIAL_CONTENT => "partial_content",
        StatusCode::MULTI_STATUS => "multi_status",
        StatusCode::ALREADY_REPORTED => "already_reported",
        StatusCode::IM_USED => "im_used",

        StatusCode::MULTIPLE_CHOICES => "multiple_choices",
        StatusCode::MOVED_PERMANENTLY => "moved_permanently",
        StatusCode::FOUND => "found",
        StatusCode::SEE_OTHER => "see_other",
        StatusCode::NOT_MODIFIED => "not_modified",
        StatusCode::USE_PROXY => "use_proxy",
        StatusCode::TEMPORARY_REDIRECT => "temporary_redirect",
        StatusCode::PERMANENT_REDIRECT => "permanent_redirect",

        StatusCode::BAD_REQUEST => "bad_request",
        StatusCode::UNAUTHORIZED => "unauthorized",
        StatusCode::PAYMENT_REQUIRED => "payment_required",
        StatusCode::FORBIDDEN => "forbidden",
        StatusCode::NOT_FOUND => "not_found",
        StatusCode::METHOD_NOT_ALLOWED => "method_not_allowed",
        StatusCode::NOT_ACCEPTABLE => "not_acceptable",
        StatusCode::PROXY_AUTHENTICATION_REQUIRED => "proxy_authentication_required",
        StatusCode::REQUEST_TIMEOUT => "request_timeout",
        StatusCode::CONFLICT => "conflict",
        StatusCode::GONE => "gone",
        StatusCode::LENGTH_REQUIRED => "length_required",
        StatusCode::PRECONDITION_FAILED => "precondition_failed",
        StatusCode::PAYLOAD_TOO_LARGE => "payload_too_large",
        StatusCode::URI_TOO_LONG => "uri_too_long",
        StatusCode::UNSUPPORTED_MEDIA_TYPE => "unsupported_media_type",
        StatusCode::RANGE_NOT_SATISFIABLE => "range_not_satisfiable",
        StatusCode::EXPECTATION_FAILED => "expectation_failed",
        StatusCode::IM_A_TEAPOT => "im_a_teapot",
        StatusCode::MISDIRECTED_REQUEST => "misdirected_request",
        StatusCode::UNPROCESSABLE_ENTITY => "unprocessable_entity",
        StatusCode::LOCKED => "locked",
        StatusCode::FAILED_DEPENDENCY => "failed_dependency",
        StatusCode::UPGRADE_REQUIRED => "upgrade_required",
        StatusCode::PRECONDITION_REQUIRED => "precondition_required",
        StatusCode::TOO_MANY_REQUESTS => "too_many_requests",
        StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE => "request_header_fields_too_large",
        StatusCode::UNAVAILABLE_FOR_LEGAL_REASONS => "unavailable_for_legal_reasons",

        StatusCode::INTERNAL_SERVER_ERROR => "internal_server_error",
        StatusCode::NOT_IMPLEMENTED => "not_implemented",
        StatusCode::BAD_GATEWAY => "bad_gateway",
        StatusCode::SERVICE_UNAVAILABLE => "service_unavailable",
        StatusCode::GATEWAY_TIMEOUT => "gateway_timeout",
        StatusCode::HTTP_VERSION_NOT_SUPPORTED => "http_version_not_supported",
        StatusCode::VARIANT_ALSO_NEGOTIATES => "variant_also_negotiates",
        StatusCode::INSUFFICIENT_STORAGE => "insufficient_storage",
        StatusCode::LOOP_DETECTED => "loop_detected",
        StatusCode::NOT_EXTENDED => "not_extended",
        StatusCode::NETWORK_AUTHENTICATION_REQUIRED => "network_authentication_required",

        _ => "unknown_status_code",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_error_string() {
        let response = internal_error("Test error");
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_internal_error_format() {
        let response = internal_error(format!("Error: {}", 42));
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_bad_request() {
        let response = bad_request("Invalid input");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_not_found() {
        let response = not_found("Resource not found");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_service_unavailable() {
        let response = service_unavailable("No workers");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
