//! Response Processing stage for responses pipeline
//!
//! This stage:
//! - Executes MCP tool loop if MCP is active (non-streaming only)
//! - OR parses non-streaming response from execution stage
//! - Records circuit breaker success/failure based on status
//! - Stores parsed JSON response

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::Value;

use super::{
    super::mcp::{execute_tool_loop, McpLoopConfig},
    ResponsesStage,
};
use crate::routers::openai::responses::{ProcessedResponse, ResponsesRequestContext};

/// Response processing stage for responses pipeline
pub struct ResponsesResponseProcessingStage;

#[async_trait]
impl ResponsesStage for ResponsesResponseProcessingStage {
    async fn execute(
        &self,
        ctx: &mut ResponsesRequestContext,
    ) -> Result<Option<Response>, Response> {
        // Get prerequisites
        let discovery = ctx.state.discovery.as_ref().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Discovery stage not completed",
            )
                .into_response()
        })?;

        let payload_output = ctx.state.payload.as_ref().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Payload building stage not completed",
            )
                .into_response()
        })?;

        let mcp_output = ctx.state.mcp.as_ref().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "MCP preparation stage not completed",
            )
                .into_response()
        })?;

        // Streaming was already handled in Stage 6, so we should never get here for streaming
        if payload_output.is_streaming {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Streaming requests should have exited in RequestExecution stage",
            )
                .into_response());
        }

        let response_json: Value = if mcp_output.active {
            // MCP tool loop path (non-streaming only)
            let url = format!("{}/v1/responses", discovery.endpoint_url);

            let mcp_config = McpLoopConfig {
                max_iterations: mcp_output.max_iterations,
            };

            let initial_payload = payload_output.json_payload.clone();

            // Execute tool loop synchronously (blocks until complete or max iterations)
            match execute_tool_loop(
                &ctx.dependencies.http_client,
                &url,
                ctx.input.headers.as_ref(),
                initial_payload,
                ctx.request(),
                &ctx.dependencies.mcp_manager,
                &mcp_config,
            )
            .await
            {
                Ok(json) => json,
                Err(e) => {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("MCP tool loop failed: {}", e),
                    )
                        .into_response());
                }
            }
        } else {
            // Non-MCP path: parse response from execution stage
            let execution = ctx.state.execution.take().ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Execution stage not completed",
                )
                    .into_response()
            })?;

            // Record circuit breaker based on status
            if execution.status.is_success() {
                ctx.dependencies.circuit_breaker.record_success();
            } else {
                ctx.dependencies.circuit_breaker.record_failure();
            }

            // Parse response JSON
            match execution.response.json::<Value>().await {
                Ok(json) => json,
                Err(e) => {
                    ctx.dependencies.circuit_breaker.record_failure();
                    return Err((
                        StatusCode::BAD_GATEWAY,
                        format!("Failed to parse upstream response: {}", e),
                    )
                        .into_response());
                }
            }
        };

        // Store processed response
        ctx.state.processed = Some(ProcessedResponse {
            json_response: response_json,
        });

        Ok(None) // Continue to persistence stage
    }

    fn name(&self) -> &'static str {
        "ResponsesResponseProcessing"
    }
}
