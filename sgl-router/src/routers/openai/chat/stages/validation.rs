//! Validation stage for chat pipeline
//!
//! This stage:
//! - Checks circuit breaker status
//! - Extracts authorization header
//! - Validates basic request parameters

use std::time::Instant;

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::ChatStage;
use crate::routers::openai::{chat::ChatRequestContext, utils::extract_auth_header};

/// Validation and authentication stage for chat pipeline
pub struct ChatValidationStage;

#[async_trait]
impl ChatStage for ChatValidationStage {
    async fn execute(&self, ctx: &mut ChatRequestContext) -> Result<(), Response> {
        // 1. Circuit breaker check
        if !ctx.dependencies.circuit_breaker.can_execute() {
            return Err((StatusCode::SERVICE_UNAVAILABLE, "Circuit breaker open").into_response());
        }

        // 2. Extract authorization header
        let auth_header = extract_auth_header(ctx.input.headers.as_ref()).map(|s| s.to_string());

        // 3. Validate model is specified
        let model = ctx.model();
        if model.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "Model parameter is required and cannot be empty",
            )
                .into_response());
        }

        // 4. Store validation output
        ctx.state.validation = Some(crate::routers::openai::chat::ValidationOutput {
            auth_header,
            validated_at: Instant::now(),
        });

        Ok(())
    }

    fn name(&self) -> &'static str {
        "ChatValidation"
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
        protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        routers::openai::chat::{ChatDependencies, ChatRequestInput},
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
    async fn test_validation_stage_success() {
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

        let dependencies = create_test_dependencies();
        let mut ctx = ChatRequestContext::new(
            Arc::new(request),
            Some(headers),
            None,
            dependencies,
        );

        let stage = ChatValidationStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());
        assert!(ctx.state.validation.is_some());

        let validation = ctx.state.validation.unwrap();
        assert_eq!(validation.auth_header, Some("Bearer test-token".to_string()));
    }

    #[tokio::test]
    async fn test_validation_stage_circuit_breaker_open() {
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            ..Default::default()
        };

        let dependencies = create_test_dependencies();

        // Open circuit breaker
        for _ in 0..10 {
            dependencies.circuit_breaker.record_failure();
        }

        let mut ctx = ChatRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        let stage = ChatValidationStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validation_stage_empty_model() {
        let request = ChatCompletionRequest {
            model: "".to_string(),
            messages: vec![],
            ..Default::default()
        };

        let dependencies = create_test_dependencies();
        let mut ctx = ChatRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        let stage = ChatValidationStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_err());
    }
}
