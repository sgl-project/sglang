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
