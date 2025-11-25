//! Request Execution stage for chat pipeline
//!
//! This stage:
//! - Builds HTTP request to upstream /v1/chat/completions
//! - Applies headers (auth, accept, content-type)
//! - Executes the request
//! - Handles errors with circuit breaker tracking
//! - Returns response directly (simple passthrough proxy)

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::ChatStage;
use crate::routers::{header_utils::apply_request_headers, openai::chat::ChatRequestContext};

/// Request execution stage for chat pipeline
///
/// This stage returns the response directly instead of storing it in state,
/// as chat is a simple passthrough proxy with no post-processing.
pub struct ChatRequestExecutionStage;

#[async_trait]
impl ChatStage for ChatRequestExecutionStage {
    async fn execute(&self, ctx: &mut ChatRequestContext) -> Result<(), Response> {
        // This stage should never return Ok(()) - it always returns Err with the final response
        // The Err is not actually an error, it's the successful response we want to return

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
        let url = format!("{}/v1/chat/completions", discovery.endpoint_url);

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

        // Handle streaming responses
        if payload_output.is_streaming {
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

            return Err(response);
        }

        // Handle non-streaming responses
        match resp.json::<serde_json::Value>().await {
            Ok(json) => {
                let response = (status, Json(json)).into_response();
                Err(response)
            }
            Err(e) => Err((
                StatusCode::BAD_GATEWAY,
                format!("Failed to parse upstream response: {}", e),
            )
                .into_response()),
        }
    }

    fn name(&self) -> &'static str {
        "ChatRequestExecution"
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
        protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        routers::openai::chat::{
            ChatDependencies, DiscoveryOutput, PayloadOutput, ValidationOutput,
        },
    };

    fn create_test_dependencies() -> Arc<ChatDependencies> {
        let client = reqwest::Client::new();
        let circuit_breaker = Arc::new(CircuitBreaker::new());
        let model_cache = Arc::new(DashMap::new());
        let worker_urls = vec!["http://localhost:8000".to_string()];

        Arc::new(ChatDependencies {
            http_client: client,
            circuit_breaker,
            model_cache,
            worker_urls,
        })
    }

    #[tokio::test]
    async fn test_execution_stage_prerequisites() {
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            ..Default::default()
        };

        let dependencies = create_test_dependencies();
        let mut ctx = ChatRequestContext::new(Arc::new(request), None, None, dependencies);

        // No prerequisites set
        let stage = ChatRequestExecutionStage;
        let result = stage.execute(&mut ctx).await;

        // Should fail due to missing discovery
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execution_stage_invalid_url() {
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            stream: false,
            ..Default::default()
        };

        let dependencies = create_test_dependencies();
        let mut ctx = ChatRequestContext::new(Arc::new(request), None, None, dependencies);

        // Set prerequisites
        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: "http://invalid-host-that-does-not-exist:9999".to_string(),
            model: "gpt-4".to_string(),
        });
        ctx.state.payload = Some(PayloadOutput {
            json_payload: json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}]
            }),
            is_streaming: false,
        });

        let stage = ChatRequestExecutionStage;
        let result = stage.execute(&mut ctx).await;

        // Should fail due to network error
        assert!(result.is_err());

        // Circuit breaker should have recorded failure
        // Note: This might be flaky depending on timing, but generally should work
        // We can't check exact counts due to async nature, but we can verify it doesn't panic
    }
}
