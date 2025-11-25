//! Request Execution stage for responses pipeline
//!
//! This stage:
//! - Builds HTTP request to upstream /v1/responses
//! - Applies headers (auth, accept, content-type)
//! - Executes the request
//! - Handles errors with circuit breaker tracking
//! - Returns early for streaming (esp. MCP streaming)
//! - Stores execution result for non-streaming

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::ResponsesStage;
use crate::routers::{
    header_utils::apply_request_headers,
    openai::responses::{ExecutionResult, ResponsesRequestContext},
};

/// Request execution stage for responses pipeline
pub struct ResponsesRequestExecutionStage;

#[async_trait]
impl ResponsesStage for ResponsesRequestExecutionStage {
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

        // Build URL
        let url = format!("{}/v1/responses", discovery.endpoint_url);

        // Build request
        let mut request_builder = ctx
            .dependencies
            .http_client
            .post(&url)
            .json(&payload_output.json_payload);

        // Apply headers
        if let Some(headers) = &ctx.input.headers {
            request_builder = apply_request_headers(headers, request_builder, true);
        }

        // Set Accept header for streaming
        if payload_output.is_streaming {
            request_builder = request_builder.header("Accept", "text/event-stream");
        }

        // Execute request
        let resp = match request_builder.send().await {
            Ok(r) => r,
            Err(e) => {
                // Record circuit breaker failure
                ctx.dependencies.circuit_breaker.record_failure();
                return Err((
                    StatusCode::BAD_GATEWAY,
                    format!("Failed to contact upstream: {}", e),
                )
                    .into_response());
            }
        };

        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Record circuit breaker success for successful status codes
        if status.is_success() {
            ctx.dependencies.circuit_breaker.record_success();
        } else {
            ctx.dependencies.circuit_breaker.record_failure();
        }

        // Handle streaming responses - return early
        if payload_output.is_streaming {
            // Check if MCP is active for this request
            let mcp_active = ctx
                .state
                .mcp
                .as_ref()
                .map(|m| m.active)
                .unwrap_or(false);

            if mcp_active {
                // Use MCP-aware streaming handler
                use crate::routers::openai::streaming::handle_streaming_response;

                let previous_response_id = ctx
                    .state
                    .context
                    .as_ref()
                    .and_then(|c| c.previous_response_id.clone());

                return Ok(Some(
                    handle_streaming_response(
                        &ctx.dependencies.http_client,
                        &ctx.dependencies.circuit_breaker,
                        Some(&ctx.dependencies.mcp_manager),
                        ctx.dependencies.response_storage.clone(),
                        ctx.dependencies.conversation_storage.clone(),
                        ctx.dependencies.conversation_item_storage.clone(),
                        url.clone(),
                        ctx.input.headers.as_ref(),
                        payload_output.json_payload.clone(),
                        ctx.request(),
                        previous_response_id,
                    )
                    .await,
                ));
            }

            // Simple passthrough for non-MCP streaming
            let stream = resp.bytes_stream();
            let (tx, rx) = mpsc::unbounded_channel();

            tokio::spawn(async move {
                let mut s = stream;
                while let Some(chunk) = s.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if tx.send(Ok(bytes)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Stream error: {}", e)));
                            break;
                        }
                    }
                }
            });

            let mut response = Response::new(Body::from_stream(UnboundedReceiverStream::new(rx)));
            *response.status_mut() = status;
            response
                .headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            // Early exit for streaming
            return Ok(Some(response));
        }

        // Non-streaming: store execution result and continue
        ctx.state.execution = Some(ExecutionResult {
            response: resp,
            status,
        });

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ResponsesRequestExecution"
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
        routers::openai::responses::{
            ContextOutput, DiscoveryOutput, PayloadOutput, ResponsesDependencies,
            ValidationOutput,
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
    async fn test_execution_stage_prerequisites() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let dependencies = create_test_dependencies().await;
        let mut ctx = ResponsesRequestContext::new(Arc::new(request), None, None, dependencies);

        // No prerequisites set
        let stage = ResponsesRequestExecutionStage;
        let result = stage.execute(&mut ctx).await;

        // Should fail due to missing discovery
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execution_stage_invalid_url() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            stream: Some(false),
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
            endpoint_url: "http://invalid-host-that-does-not-exist:9999".to_string(),
            model: "gpt-4".to_string(),
        });
        ctx.state.context = Some(ContextOutput {
            conversation_items: None,
            conversation_id: None,
            previous_response_id: None,
        });
        ctx.state.payload = Some(PayloadOutput {
            json_payload: json!({
                "model": "gpt-4",
                "input": "Hello"
            }),
            is_streaming: false,
        });

        let stage = ResponsesRequestExecutionStage;
        let result = stage.execute(&mut ctx).await;

        // Should fail due to network error
        assert!(result.is_err());
    }
}
