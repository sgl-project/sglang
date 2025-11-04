//! Shared response handlers for both regular and harmony implementations
//!
//! These handlers are used by both pipelines for retrieving and cancelling responses.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;
use tracing::{debug, error, warn};

use crate::{
    data_connector::ResponseId, routers::grpc::regular::responses::context::ResponsesContext,
};

/// Implementation for GET /v1/responses/{response_id}
///
/// Retrieves a stored response from the database.
/// Used by both regular and harmony implementations.
pub async fn get_response_impl(ctx: &ResponsesContext, response_id: &str) -> Response {
    let resp_id = ResponseId::from(response_id);

    // Retrieve response from storage
    match ctx.response_storage.get_response(&resp_id).await {
        Ok(Some(stored_response)) => axum::Json(stored_response.raw_response).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            axum::Json(json!({
                "error": {
                    "message": format!("Response with id '{}' not found", response_id),
                    "type": "not_found_error",
                    "code": "response_not_found"
                }
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({
                "error": {
                    "message": format!("Failed to retrieve response: {}", e),
                    "type": "internal_error"
                }
            })),
        )
            .into_response(),
    }
}

/// Implementation for POST /v1/responses/{response_id}/cancel
///
/// Cancels a background response if it's still in progress.
pub async fn cancel_response_impl(ctx: &ResponsesContext, response_id: &str) -> Response {
    let resp_id = ResponseId::from(response_id);

    // Retrieve response from storage to check if it exists and get current status
    match ctx.response_storage.get_response(&resp_id).await {
        Ok(Some(stored_response)) => {
            // Check current status - only queued or in_progress responses can be cancelled
            let current_status = stored_response
                .raw_response
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            match current_status {
                "queued" | "in_progress" => {
                    // Attempt to abort the background task
                    let mut tasks = ctx.background_tasks.write().await;
                    if let Some(task_info) = tasks.remove(response_id) {
                        // Abort the Rust task immediately
                        task_info.handle.abort();

                        // Abort the Python/scheduler request via gRPC (if client is available)
                        let client_opt = task_info.client.read().await;
                        if let Some(ref client) = *client_opt {
                            if let Err(e) = client
                                .abort_request(
                                    task_info.grpc_request_id.clone(),
                                    "User cancelled via API".to_string(),
                                )
                                .await
                            {
                                warn!(
                                    "Failed to abort Python request {}: {}",
                                    task_info.grpc_request_id, e
                                );
                            } else {
                                debug!(
                                    "Successfully aborted Python request: {}",
                                    task_info.grpc_request_id
                                );
                            }
                        } else {
                            debug!("Client not yet available for abort, request may not have started yet");
                        }

                        // Task was found and aborted
                        (
                            StatusCode::OK,
                            axum::Json(json!({
                                "id": response_id,
                                "status": "cancelled",
                                "message": "Background task has been cancelled"
                            })),
                        )
                            .into_response()
                    } else {
                        // Task handle not found but status is queued/in_progress
                        // This can happen if: (1) task crashed, or (2) storage persistence failed
                        error!(
                            "Response {} has status '{}' but task handle is missing. Task may have crashed or storage update failed.",
                            response_id, current_status
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            axum::Json(json!({
                                "error": {
                                    "message": "Internal error: background task completed but failed to update status in storage",
                                    "type": "internal_error",
                                    "code": "status_update_failed"
                                }
                            })),
                        )
                            .into_response()
                    }
                }
                "completed" => (
                    StatusCode::BAD_REQUEST,
                    axum::Json(json!({
                        "error": {
                            "message": "Cannot cancel completed response",
                            "type": "invalid_request_error",
                            "code": "response_already_completed"
                        }
                    })),
                )
                    .into_response(),
                "failed" => (
                    StatusCode::BAD_REQUEST,
                    axum::Json(json!({
                        "error": {
                            "message": "Cannot cancel failed response",
                            "type": "invalid_request_error",
                            "code": "response_already_failed"
                        }
                    })),
                )
                    .into_response(),
                "cancelled" => (
                    StatusCode::OK,
                    axum::Json(json!({
                        "id": response_id,
                        "status": "cancelled",
                        "message": "Response was already cancelled"
                    })),
                )
                    .into_response(),
                _ => {
                    // Unknown status
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        axum::Json(json!({
                            "error": {
                                "message": format!("Unknown response status: {}", current_status),
                                "type": "internal_error"
                            }
                        })),
                    )
                        .into_response()
                }
            }
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            axum::Json(json!({
                "error": {
                    "message": format!("Response with id '{}' not found", response_id),
                    "type": "not_found_error",
                    "code": "response_not_found"
                }
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({
                "error": {
                    "message": format!("Failed to retrieve response: {}", e),
                    "type": "internal_error"
                }
            })),
        )
            .into_response(),
    }
}
