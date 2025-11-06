//! Centralized error response handling for all routers
//!
//! This module provides consistent error responses across OpenAI and gRPC routers,
//! ensuring all errors follow OpenAI's API error format.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use tracing::{error, warn};

/// Create a 500 Internal Server Error response
///
/// Use this for unexpected server-side errors, database failures, etc.
///
/// # Example
/// ```ignore
/// return Err(internal_error("Database connection failed"));
/// ```
pub fn internal_error(message: impl Into<String>) -> Response {
    let msg = message.into();
    error!("{}", msg);
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": {
                "message": msg,
                "type": "internal_error",
                "code": 500
            }
        })),
    )
        .into_response()
}

/// Create a 400 Bad Request response
///
/// Use this for invalid request parameters, malformed JSON, validation errors, etc.
///
/// # Example
/// ```ignore
/// return Err(bad_request("Invalid conversation ID format"));
/// ```
pub fn bad_request(message: impl Into<String>) -> Response {
    let msg = message.into();
    error!("{}", msg);
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "message": msg,
                "type": "invalid_request_error",
                "code": 400
            }
        })),
    )
        .into_response()
}

/// Create a 404 Not Found response
///
/// Use this for resources that don't exist (conversations, responses, etc.)
///
/// # Example
/// ```ignore
/// return Err(not_found(format!("Conversation '{}' not found", id)));
/// ```
pub fn not_found(message: impl Into<String>) -> Response {
    let msg = message.into();
    warn!("{}", msg);
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "message": msg,
                "type": "invalid_request_error",
                "code": 404
            }
        })),
    )
        .into_response()
}

/// Create a 503 Service Unavailable response
///
/// Use this for temporary service issues like no workers available, rate limiting, etc.
///
/// # Example
/// ```ignore
/// return Err(service_unavailable("No workers available for this model"));
/// ```
pub fn service_unavailable(message: impl Into<String>) -> Response {
    let msg = message.into();
    warn!("{}", msg);
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": {
                "message": msg,
                "type": "service_unavailable",
                "code": 503
            }
        })),
    )
        .into_response()
}

/// Create a 424 Failed Dependency response
///
/// Use this when an external dependency (like MCP server) fails.
///
/// # Example
/// ```ignore
/// return Err(failed_dependency("Failed to connect to MCP server"));
/// ```
pub fn failed_dependency(message: impl Into<String>) -> Response {
    let msg = message.into();
    warn!("{}", msg);
    (
        StatusCode::FAILED_DEPENDENCY,
        Json(json!({
            "error": {
                "message": msg,
                "type": "external_connector_error",
                "code": 424
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
