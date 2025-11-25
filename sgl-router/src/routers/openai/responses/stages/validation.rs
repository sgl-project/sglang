//! Validation stage for responses pipeline
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

use super::ResponsesStage;
use crate::routers::openai::{responses::ResponsesRequestContext, utils::extract_auth_header};

/// Validation and authentication stage for responses pipeline
pub struct ResponsesValidationStage;

#[async_trait]
impl ResponsesStage for ResponsesValidationStage {
    async fn execute(
        &self,
        ctx: &mut ResponsesRequestContext,
    ) -> Result<Option<Response>, Response> {
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
        ctx.state.validation = Some(crate::routers::openai::responses::ValidationOutput {
            auth_header,
            validated_at: Instant::now(),
        });

        Ok(None) // Continue to next stage
    }

    fn name(&self) -> &'static str {
        "ResponsesValidation"
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
        protocols::responses::{ResponseInput, ResponsesRequest},
        routers::openai::responses::ResponsesDependencies,
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
    async fn test_validation_stage_success() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static("Bearer test-token"),
        );

        let dependencies = create_test_dependencies().await;
        let mut ctx = ResponsesRequestContext::new(
            Arc::new(request),
            Some(headers),
            None,
            dependencies,
        );

        let stage = ResponsesValidationStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // Should continue to next stage
        assert!(ctx.state.validation.is_some());

        let validation = ctx.state.validation.unwrap();
        assert_eq!(validation.auth_header, Some("Bearer test-token".to_string()));
    }

    #[tokio::test]
    async fn test_validation_stage_circuit_breaker_open() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let dependencies = create_test_dependencies().await;

        // Open circuit breaker
        for _ in 0..10 {
            dependencies.circuit_breaker.record_failure();
        }

        let mut ctx = ResponsesRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        let stage = ResponsesValidationStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validation_stage_empty_model() {
        let request = ResponsesRequest {
            model: "".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        };

        let dependencies = create_test_dependencies().await;
        let mut ctx = ResponsesRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        let stage = ResponsesValidationStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_err());
    }
}
