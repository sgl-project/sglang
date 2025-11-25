//! Validation & Auth stage
//!
//! This stage:
//! - Checks circuit breaker status
//! - Extracts authorization header
//! - Validates basic request parameters

use std::time::Instant;

use async_trait::async_trait;
use axum::{http::StatusCode, response::{IntoResponse, Response}};

use super::PipelineStage;
use crate::routers::openai::{
    context::{RequestContext, RequestType, ValidationOutput},
    utils::extract_auth_header,
};

/// Validation and authentication stage
pub struct ValidationStage;

#[async_trait]
impl PipelineStage for ValidationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // 1. Circuit breaker check
        if !ctx.components.circuit_breaker.can_execute() {
            return Err((StatusCode::SERVICE_UNAVAILABLE, "Circuit breaker open").into_response());
        }

        // 2. Extract authorization header
        let auth_header = extract_auth_header(ctx.input.headers.as_ref()).map(|s| s.to_string());

        // 3. Validate model is specified
        let model = match &ctx.input.request_type {
            RequestType::Chat(req) => &req.model,
            RequestType::Responses(req) => ctx.input.model_id.as_deref().unwrap_or(&req.model),
        };

        if model.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "Model parameter is required and cannot be empty",
            )
                .into_response());
        }

        // 4. Store validation output
        ctx.state.validation = Some(ValidationOutput {
            auth_header,
            validated_at: Instant::now(),
        });

        Ok(None) // Continue to next stage
    }

    fn name(&self) -> &'static str {
        "Validation"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::http::{HeaderMap, HeaderValue};
    use dashmap::DashMap;

    use super::*;
    use crate::{
        core::CircuitBreaker,
        data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        },
        mcp::{config::McpConfig, McpManager},
        protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        routers::openai::context::SharedComponents,
    };

    async fn create_test_components() -> Arc<SharedComponents> {
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
            worker_urls: vec!["http://localhost:8000".to_string()],
        })
    }

    #[tokio::test]
    async fn test_validation_stage_success() {
        let components = create_test_components().await;
        let stage = ValidationStage;

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            ..Default::default()
        };

        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer test-token"),
        );

        let mut ctx = RequestContext {
            input: crate::routers::openai::context::RequestInput {
                request_type: RequestType::Chat(Arc::new(request)),
                headers: Some(headers),
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // Should continue to next stage

        // Verify validation output was stored
        let validation = ctx.state.validation.as_ref().unwrap();
        assert_eq!(validation.auth_header.as_deref(), Some("Bearer test-token"));
    }

    #[tokio::test]
    async fn test_validation_stage_circuit_breaker_open() {
        let components = create_test_components().await;

        // Trigger circuit breaker to open by recording multiple failures
        // Default threshold is typically 5 failures
        for _ in 0..10 {
            components.circuit_breaker.record_failure();
        }

        // Verify circuit breaker is open
        assert!(!components.circuit_breaker.can_execute());

        let stage = ValidationStage;

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: crate::routers::openai::context::RequestInput {
                request_type: RequestType::Chat(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_err());

        let response = result.unwrap_err();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_validation_stage_empty_model() {
        let components = create_test_components().await;
        let stage = ValidationStage;

        let request = ChatCompletionRequest {
            model: "".to_string(), // Empty model
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: crate::routers::openai::context::RequestInput {
                request_type: RequestType::Chat(Arc::new(request)),
                headers: None,
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_err());

        let response = result.unwrap_err();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_validation_stage_no_auth_header() {
        let components = create_test_components().await;
        let stage = ValidationStage;

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            ..Default::default()
        };

        let mut ctx = RequestContext {
            input: crate::routers::openai::context::RequestInput {
                request_type: RequestType::Chat(Arc::new(request)),
                headers: None, // No headers
                model_id: None,
            },
            components,
            state: Default::default(),
        };

        let result = stage.execute(&mut ctx).await;
        assert!(result.is_ok());

        // Verify validation output was stored with None auth
        let validation = ctx.state.validation.as_ref().unwrap();
        assert!(validation.auth_header.is_none());
    }
}
