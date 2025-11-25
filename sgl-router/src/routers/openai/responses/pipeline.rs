//! Responses pipeline orchestrator
//!
//! This module provides the full 8-stage pipeline for /v1/responses:
//! 1. Validation - Auth and circuit breaker checks
//! 2. ModelDiscovery - Find endpoint URL
//! 3. HistoryLoading - Load conversation history
//! 4. RequestBuilding - Build payload with history
//! 5. McpPreparation - Setup MCP tool loop
//! 6. RequestExecution - Execute HTTP request
//! 7. ResponseProcessing - Parse response, MCP tool loop
//! 8. Persistence - Save response and conversation
//!
//! Unlike chat pipeline, responses includes MCP and persistence.

use std::sync::Arc;

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::{debug, error};

use super::{
    stages::{
        ResponsesHistoryLoadingStage, ResponsesMcpPreparationStage, ResponsesModelDiscoveryStage,
        ResponsesPersistenceStage, ResponsesRequestBuildingStage, ResponsesRequestExecutionStage,
        ResponsesResponseProcessingStage, ResponsesStage, ResponsesValidationStage,
    },
    ResponsesDependencies, ResponsesRequestContext,
};
use crate::protocols::responses::ResponsesRequest;

/// Responses pipeline for /v1/responses
///
/// This pipeline is full-featured with 8 stages:
/// - Validation, ModelDiscovery, HistoryLoading, RequestBuilding
/// - McpPreparation, RequestExecution, ResponseProcessing, Persistence
pub struct ResponsesPipeline {
    dependencies: Arc<ResponsesDependencies>,
    stages: Vec<Box<dyn ResponsesStage>>,
}

impl ResponsesPipeline {
    /// Create a new responses pipeline with dependencies
    pub fn new(dependencies: Arc<ResponsesDependencies>) -> Self {
        let stages: Vec<Box<dyn ResponsesStage>> = vec![
            Box::new(ResponsesValidationStage),
            Box::new(ResponsesModelDiscoveryStage),
            Box::new(ResponsesHistoryLoadingStage),
            Box::new(ResponsesRequestBuildingStage),
            Box::new(ResponsesMcpPreparationStage),
            Box::new(ResponsesRequestExecutionStage),
            Box::new(ResponsesResponseProcessingStage),
            Box::new(ResponsesPersistenceStage),
        ];

        Self {
            dependencies,
            stages,
        }
    }

    /// Execute the pipeline for a responses request
    ///
    /// Returns the HTTP response (streaming, non-streaming, or error)
    pub async fn execute(
        &self,
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
    ) -> Response {
        // Create request context
        let mut ctx = ResponsesRequestContext::new(
            request,
            headers,
            model_id,
            Arc::clone(&self.dependencies),
        );

        // Execute stages sequentially
        for (i, stage) in self.stages.iter().enumerate() {
            debug!("Executing responses stage {}: {}", i + 1, stage.name());

            match stage.execute(&mut ctx).await {
                Ok(None) => {
                    // Stage succeeded, continue to next stage
                    debug!("Responses stage {} completed successfully", stage.name());
                }
                Ok(Some(response)) => {
                    // Stage returned early response (streaming or final response)
                    debug!(
                        "Responses stage {} returned response, pipeline complete",
                        stage.name()
                    );
                    return response;
                }
                Err(response) => {
                    // Stage failed
                    error!("Responses stage {} failed", stage.name());
                    return response;
                }
            }
        }

        // This should never happen (Persistence always returns Some(Response))
        error!("Responses pipeline completed without returning response");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Pipeline completed without response",
        )
            .into_response()
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
        protocols::responses::{ResponseInput, ResponsesRequest},
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
    async fn test_pipeline_creation() {
        let dependencies = create_test_dependencies().await;
        let pipeline = ResponsesPipeline::new(dependencies);

        // Should have 8 stages
        assert_eq!(pipeline.stages.len(), 8);
    }

    #[tokio::test]
    async fn test_pipeline_empty_model() {
        let dependencies = create_test_dependencies().await;
        let pipeline = ResponsesPipeline::new(dependencies);

        let request = Arc::new(ResponsesRequest {
            model: "".to_string(), // Empty model should fail validation
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        });

        let response = pipeline.execute(request, None, None).await;

        // Should return error response (400 Bad Request)
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_pipeline_circuit_breaker_open() {
        let dependencies = create_test_dependencies().await;

        // Open circuit breaker
        for _ in 0..10 {
            dependencies.circuit_breaker.record_failure();
        }

        let pipeline = ResponsesPipeline::new(dependencies);

        let request = Arc::new(ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("Hello".to_string()),
            ..Default::default()
        });

        let response = pipeline.execute(request, None, None).await;

        // Should return service unavailable
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
