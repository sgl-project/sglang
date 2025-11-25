//! Response Processing stage
//!
//! This stage:
//! - Executes MCP tool loop if MCP is active
//! - OR parses non-streaming response from execution stage
//! - Records circuit breaker success/failure
//! - Stores parsed JSON response

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

use super::PipelineStage;
use crate::routers::openai::{
    context::{ProcessedResponse, RequestContext, RequestType},
    mcp::{execute_tool_loop, McpLoopConfig},
};

/// Response processing stage
pub struct ResponseProcessingStage;

#[async_trait]
impl PipelineStage for ResponseProcessingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
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
            // MCP tool loop path
            let url = match &ctx.input.request_type {
                RequestType::Responses(_) => {
                    format!("{}/v1/responses", discovery.endpoint_url)
                }
                RequestType::Chat(_) => {
                    // MCP should only be active for Responses requests
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "MCP should not be active for Chat requests",
                    )
                        .into_response());
                }
            };

            let responses_req = match &ctx.input.request_type {
                RequestType::Responses(req) => req,
                _ => unreachable!(),
            };

            let config = McpLoopConfig {
                max_iterations: mcp_output.max_iterations,
            };

            match execute_tool_loop(
                &ctx.components.http_client,
                &url,
                ctx.input.headers.as_ref(),
                payload_output.json_payload.clone(),
                responses_req,
                &ctx.components.mcp_manager,
                &config,
            )
            .await
            {
                Ok(resp) => resp,
                Err(err) => {
                    ctx.components.circuit_breaker.record_failure();
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(json!({"error": {"message": err}})),
                    )
                        .into_response());
                }
            }
        } else {
            // Simple request path - take execution result (consumes response)
            let execution = ctx.state.execution.take().ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Execution stage not completed (non-MCP path requires ExecutionResult)",
                )
                    .into_response()
            })?;

            // Check status
            if !execution.status.is_success() {
                ctx.components.circuit_breaker.record_failure();
                let body = execution
                    .response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Failed to read error response".to_string());
                return Err((execution.status, body).into_response());
            }

            // Parse JSON
            match execution.response.json::<Value>().await {
                Ok(json) => json,
                Err(e) => {
                    ctx.components.circuit_breaker.record_failure();
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to parse upstream response: {}", e),
                    )
                        .into_response());
                }
            }
        };

        // Record success
        ctx.components.circuit_breaker.record_success();

        // Store processed response
        ctx.state.processed = Some(ProcessedResponse {
            json_response: response_json,
        });

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ResponseProcessing"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dashmap::DashMap;

    use super::*;
    use crate::{
        core::CircuitBreaker,
        data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        },
        mcp::{config::McpConfig, McpManager},
        protocols::{
            chat::{ChatCompletionRequest, ChatMessage, MessageContent},
            responses::{ResponseInput, ResponsesRequest},
        },
        routers::openai::{
            context::{DiscoveryOutput, McpOutput, PayloadOutput, RequestInput, SharedComponents},
            mcp::ToolLoopState,
        },
    };

    async fn create_test_components(worker_urls: Vec<String>) -> Arc<SharedComponents> {
        let client = reqwest::Client::new();
        let circuit_breaker = Arc::new(CircuitBreaker::new());
        let model_cache = Arc::new(DashMap::new());

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

        Arc::new(SharedComponents {
            http_client: client,
            circuit_breaker,
            model_cache,
            mcp_manager,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            worker_urls,
        })
    }

    #[tokio::test]
    async fn test_response_processing_stage_prerequisites() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = ResponseProcessingStage;

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: RequestInput {
                request_type: RequestType::Chat(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        // Without prerequisites, should error
        let result = stage.execute(&mut ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_response_processing_stage_streaming_check() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = ResponseProcessingStage;

        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: RequestInput {
                request_type: RequestType::Responses(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        // Set up prerequisites with streaming=true
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: "http://test:8000".to_string(),
            model: "gpt-4".to_string(),
        });
        ctx.state.payload = Some(PayloadOutput {
            json_payload: serde_json::json!({"model": "gpt-4"}),
            is_streaming: true, // Streaming should have exited in Stage 6
        });
        ctx.state.mcp = Some(McpOutput {
            active: false,
            tool_loop_state: ToolLoopState {
                iteration: 0,
                total_calls: 0,
                conversation_history: vec![],
                original_input: ResponseInput::Text(String::new()),
            },
            max_iterations: 0,
        });

        // Should error because streaming requests shouldn't reach this stage
        let result = stage.execute(&mut ctx).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_response_processing_stage_mcp_inactive_no_execution() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = ResponseProcessingStage;

        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: RequestInput {
                request_type: RequestType::Responses(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        // Set up prerequisites but no execution result
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: "http://test:8000".to_string(),
            model: "gpt-4".to_string(),
        });
        ctx.state.payload = Some(PayloadOutput {
            json_payload: serde_json::json!({"model": "gpt-4"}),
            is_streaming: false,
        });
        ctx.state.mcp = Some(McpOutput {
            active: false,
            tool_loop_state: ToolLoopState {
                iteration: 0,
                total_calls: 0,
                conversation_history: vec![],
                original_input: ResponseInput::Text(String::new()),
            },
            max_iterations: 0,
        });
        // No execution result set

        // Should error because non-MCP path requires execution result
        let result = stage.execute(&mut ctx).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
