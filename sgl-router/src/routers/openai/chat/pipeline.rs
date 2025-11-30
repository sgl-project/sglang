//! Chat pipeline orchestrator
//!
//! This module provides the lightweight 4-stage pipeline for /v1/chat/completions:
//! 1. Validation - Auth and circuit breaker checks
//! 2. ModelDiscovery - Find endpoint URL
//! 3. RequestBuilding - Strip SGLang fields, transformations
//! 4. RequestExecution - Execute HTTP request and return response
//!
//! Unlike the responses pipeline, chat is a simple proxy with no MCP or persistence.

use std::sync::Arc;

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::{debug, error};

use super::{
    stages::{
        ChatModelDiscoveryStage, ChatRequestBuildingStage, ChatRequestExecutionStage, ChatStage,
        ChatValidationStage,
    },
    ChatDependencies, ChatRequestContext,
};
use crate::protocols::chat::ChatCompletionRequest;

/// Chat pipeline for /v1/chat/completions
///
/// This pipeline is a lightweight proxy with 4 stages:
/// - Validation: Auth and circuit breaker
/// - ModelDiscovery: Find endpoint
/// - RequestBuilding: Build JSON payload
/// - RequestExecution: Execute and return response
pub struct ChatPipeline {
    dependencies: Arc<ChatDependencies>,
    stages: Vec<Box<dyn ChatStage>>,
}

impl ChatPipeline {
    /// Create a new chat pipeline with dependencies
    pub fn new(dependencies: Arc<ChatDependencies>) -> Self {
        let stages: Vec<Box<dyn ChatStage>> = vec![
            Box::new(ChatValidationStage),
            Box::new(ChatModelDiscoveryStage),
            Box::new(ChatRequestBuildingStage),
            Box::new(ChatRequestExecutionStage),
        ];

        Self {
            dependencies,
            stages,
        }
    }

    /// Execute the pipeline for a chat completion request
    ///
    /// Returns the HTTP response (either streaming or JSON)
    pub async fn execute(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
    ) -> Response {
        // Create request context
        let mut ctx =
            ChatRequestContext::new(request, headers, model_id, Arc::clone(&self.dependencies));

        // Execute stages sequentially
        for (i, stage) in self.stages.iter().enumerate() {
            debug!("Executing chat stage {}: {}", i + 1, stage.name());

            match stage.execute(&mut ctx).await {
                Ok(()) => {
                    // Stage succeeded, continue to next stage
                    debug!("Chat stage {} completed successfully", stage.name());
                }
                Err(response) => {
                    // Stage failed or returned final response
                    // For RequestExecution stage, this is the expected behavior (returns response)
                    if i == self.stages.len() - 1 {
                        // Last stage (RequestExecution) - this is the successful response
                        debug!("Chat pipeline completed, returning response");
                        return response;
                    } else {
                        // Earlier stage error - log and return error
                        error!("Chat stage {} failed", stage.name());
                        return response;
                    }
                }
            }
        }

        // This should never happen (RequestExecution always returns Err with response)
        error!("Chat pipeline completed without returning response");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Pipeline completed without response",
        )
            .into_response()
    }
}
