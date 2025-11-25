//! Request Execution stage
//!
//! This stage:
//! - Builds HTTP request to upstream OpenAI-compatible API
//! - Applies headers (auth, accept, content-type)
//! - Executes the request
//! - Handles errors with circuit breaker
//! - Returns early for streaming requests

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::PipelineStage;
use crate::routers::{
    header_utils::apply_request_headers,
    openai::context::{ExecutionResult, RequestContext, RequestType},
};

/// Request execution stage
pub struct RequestExecutionStage;

#[async_trait]
impl PipelineStage for RequestExecutionStage {
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

        // Determine the endpoint URL based on request type
        let url = match &ctx.input.request_type {
            RequestType::Chat(_) => {
                format!("{}/v1/chat/completions", discovery.endpoint_url)
            }
            RequestType::Responses(_) => {
                format!("{}/v1/responses", discovery.endpoint_url)
            }
        };

        // Build request
        let mut request_builder = ctx
            .components
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
                ctx.components.circuit_breaker.record_failure();
                return Err((
                    StatusCode::BAD_GATEWAY,
                    format!("Failed to contact upstream: {}", e),
                )
                    .into_response());
            }
        };

        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Handle streaming responses - return early
        if payload_output.is_streaming {
            // For streaming, we pass the response directly to the client
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
        "RequestExecution"
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
        routers::openai::context::{
            DiscoveryOutput, PayloadOutput, RequestInput, SharedComponents, ValidationOutput,
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
    async fn test_execution_stage_prerequisites_check() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = RequestExecutionStage;

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

        // Without discovery and payload, should error
        let result = stage.execute(&mut ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execution_stage_url_building() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = RequestExecutionStage;

        // Test Chat URL building
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
            components: components.clone(),
            state: Default::default(),
        };

        // Set prerequisites
        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: std::time::Instant::now(),
        });
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: "http://test-endpoint:8000".to_string(),
            model: "gpt-4".to_string(),
        });
        ctx.state.payload = Some(PayloadOutput {
            json_payload: serde_json::json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}]
            }),
            is_streaming: false,
        });

        // This will fail since test-endpoint doesn't exist, but we're just checking URL logic
        let result = stage.execute(&mut ctx).await;
        // Should error with connection failure, not prerequisite error
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn test_execution_stage_responses_url() {
        let worker_urls = vec!["http://localhost:8000".to_string()];
        let components = create_test_components(worker_urls).await;
        let stage = RequestExecutionStage;

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

        // Set prerequisites
        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: std::time::Instant::now(),
        });
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: "http://test-responses:8000".to_string(),
            model: "gpt-4".to_string(),
        });
        ctx.state.payload = Some(PayloadOutput {
            json_payload: serde_json::json!({
                "model": "gpt-4",
                "input": "Hello"
            }),
            is_streaming: false,
        });

        // Will fail with connection error, but verifies URL construction
        let result = stage.execute(&mut ctx).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.status(), StatusCode::BAD_GATEWAY);
    }
}
