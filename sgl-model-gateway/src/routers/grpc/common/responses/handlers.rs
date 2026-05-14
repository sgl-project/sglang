//! Shared response handlers for both regular and harmony implementations
//!
//! These handlers are used by both pipelines for retrieving and cancelling responses.

use axum::response::{IntoResponse, Response};

use super::ResponsesContext;
use crate::{data_connector::ResponseId, routers::error};

/// Implementation for GET /v1/responses/{response_id}
///
/// Retrieves a stored response from the database.
/// Used by both regular and harmony implementations.
pub(crate) async fn get_response_impl(ctx: &ResponsesContext, response_id: &str) -> Response {
    let resp_id = ResponseId::from(response_id);

    // Retrieve response from storage
    match ctx.response_storage.get_response(&resp_id).await {
        Ok(Some(stored_response)) => axum::Json(stored_response.raw_response).into_response(),
        Ok(None) => error::not_found(
            "response_not_found",
            format!("Response with id '{}' not found", response_id),
        ),
        Err(e) => error::internal_error(
            "retrieve_response_failed",
            format!("Failed to retrieve response: {}", e),
        ),
    }
}

/// Implementation for POST /v1/responses/{response_id}/cancel
///
/// Background mode is no longer supported, so this endpoint always returns
/// an error indicating that cancellation is not available.
pub(crate) async fn cancel_response_impl(ctx: &ResponsesContext, response_id: &str) -> Response {
    let resp_id = ResponseId::from(response_id);

    // Check if response exists
    match ctx.response_storage.get_response(&resp_id).await {
        Ok(Some(stored_response)) => {
            let current_status = stored_response
                .raw_response
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            match current_status {
                "completed" => error::bad_request(
                    "response_already_completed",
                    "Cannot cancel completed response",
                ),
                "failed" => {
                    error::bad_request("response_already_failed", "Cannot cancel failed response")
                }
                _ => {
                    // Background mode is no longer supported, so there's nothing to cancel
                    error::bad_request(
                        "cancellation_not_supported",
                        "Background mode is not supported. Synchronous and streaming responses cannot be cancelled.",
                    )
                }
            }
        }
        Ok(None) => error::not_found(
            "response_not_found",
            format!("Response with id '{}' not found", response_id),
        ),
        Err(e) => error::internal_error(
            "retrieve_response_failed",
            format!("Failed to retrieve response: {}", e),
        ),
    }
}
