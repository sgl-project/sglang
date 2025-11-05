//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

use crate::{
    core::WorkerRegistry,
    mcp::McpManager,
    protocols::responses::{ResponseTool, ResponseToolType},
    routers::{grpc::error, openai::mcp::ensure_request_mcp_client},
};

/// Ensure MCP connection succeeds if MCP tools are declared
///
/// Checks if request declares MCP tools, and if so, validates that
/// the MCP client can be created and connected.
pub async fn ensure_mcp_connection(
    mcp_manager: &Arc<McpManager>,
    tools: Option<&[ResponseTool]>,
) -> Result<bool, Response> {
    let has_mcp_tools = tools
        .map(|t| {
            t.iter()
                .any(|tool| matches!(tool.r#type, ResponseToolType::Mcp))
        })
        .unwrap_or(false);

    if has_mcp_tools {
        if let Some(tools) = tools {
            if ensure_request_mcp_client(mcp_manager, tools)
                .await
                .is_none()
            {
                return Err(error::failed_dependency(
                    "Failed to connect to MCP server. Check server_url and authorization.",
                ));
            }
        }
    }

    Ok(has_mcp_tools)
}

/// Validate that workers are available for the requested model
pub fn validate_worker_availability(
    worker_registry: &Arc<WorkerRegistry>,
    model: &str,
) -> Option<Response> {
    let available_models = worker_registry.get_models();

    if !available_models.contains(&model.to_string()) {
        return Some(
            (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(json!({
                    "error": {
                        "message": format!(
                            "No workers available for model '{}'. Available models: {}",
                            model,
                            available_models.join(", ")
                        ),
                        "type": "service_unavailable",
                        "param": "model",
                        "code": "no_available_workers"
                    }
                })),
            )
                .into_response(),
        );
    }

    None
}
