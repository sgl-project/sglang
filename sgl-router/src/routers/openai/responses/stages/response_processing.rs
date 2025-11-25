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

use super::ResponsesStage;
use crate::routers::openai::{
    mcp::{execute_tool_loop, McpLoopConfig},
    responses::{ProcessedResponse, ResponsesRequestContext},
};

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
            ).await {
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
        ctx.state.processed = Some(ProcessedResponse { json_response: response_json });

        Ok(None) // Continue to persistence stage
    }

    fn name(&self) -> &'static str {
        "ResponsesResponseProcessing"
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

    use dashmap::DashMap;
    use serde_json::json;

    use super::*;
    use crate::{
        core::CircuitBreaker,
        data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        },
        mcp::{config::McpConfig, McpManager},
        protocols::responses::{ResponseInput, ResponsesRequest},
        routers::openai::{
            mcp::ToolLoopState,
            responses::{
                ContextOutput, DiscoveryOutput, McpOutput, PayloadOutput, ResponsesDependencies,
                ValidationOutput,
            },
        },
    };

    async fn create_test_dependencies() -> Arc<ResponsesDependencies> {
        let client = reqwest::Client::new();
        let circuit_breaker = Arc::new(CircuitBreaker::new());
        let model_cache = Arc::new(DashMap::new());
        let worker_urls = vec!["http://localhost:8000".to_string()];

        let mcp_config = McpConfig {
            servers: vec![],
            pool: Default::default(),
            proxy: None,
            warmup: vec![],
            inventory: Default::default(),
        };
        let mcp_manager = Arc::new(
            McpManager::new(mcp_config, 10)
                .await
                .expect("Failed to create MCP manager"),
        );

        let response_storage = Arc::new(MemoryResponseStorage::new());
        let conversation_storage = Arc::new(MemoryConversationStorage::new());
        let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

        Arc::new(ResponsesDependencies {
            http_client: client,
            circuit_breaker,
            model_cache,
            worker_urls,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            mcp_manager,
        })
    }

    #[tokio::test]
    async fn test_response_processing_stage_prerequisites() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let dependencies = create_test_dependencies().await;
        let mut ctx = ResponsesRequestContext::new(Arc::new(request), None, None, dependencies);

        // No prerequisites set
        let stage = ResponsesResponseProcessingStage;
        let result = stage.execute(&mut ctx).await;

        // Should fail due to missing prerequisites
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_response_processing_stage_streaming_error() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            stream: Some(true),
            ..Default::default()
        };

        let dependencies = create_test_dependencies().await;
        let mut ctx = ResponsesRequestContext::new(Arc::new(request), None, None, dependencies);

        // Set prerequisites
        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: "http://localhost:8000".to_string(),
            model: "gpt-4".to_string(),
        });
        ctx.state.context = Some(ContextOutput {
            conversation_items: None,
            conversation_id: None,
            previous_response_id: None,
        });
        ctx.state.payload = Some(PayloadOutput {
            json_payload: json!({"model": "gpt-4", "input": "Hello"}),
            is_streaming: true, // Streaming should have been handled earlier
        });
        ctx.state.mcp = Some(McpOutput {
            active: false,
            tool_loop_state: ToolLoopState {
                iteration: 0,
                total_calls: 0,
                conversation_history: vec![],
                original_input: ResponseInput::Text("Hello".to_string()),
            },
            max_iterations: 0,
        });

        let stage = ResponsesResponseProcessingStage;
        let result = stage.execute(&mut ctx).await;

        // Should error because streaming should have been handled in RequestExecution
        assert!(result.is_err());
    }
}
