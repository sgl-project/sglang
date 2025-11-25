//! Pipeline orchestrator for OpenAI router request processing
//!
//! This module defines the RequestPipeline orchestrator that coordinates
//! the execution of pipeline stages from request preparation to response delivery.

use std::sync::Arc;

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use tracing::error;

use super::{
    context::*,
    stages::{
        ContextLoadingStage, McpPreparationStage, ModelDiscoveryStage, PersistenceStage,
        PipelineStage, RequestBuildingStage, RequestExecutionStage, ResponseProcessingStage,
        ValidationStage,
    },
};

/// Generic request pipeline for all request types
///
/// Orchestrates all stages from request preparation to response delivery.
#[derive(Clone)]
pub struct RequestPipeline {
    stages: Arc<Vec<Box<dyn PipelineStage>>>,
}

impl RequestPipeline {
    /// Get the number of stages in this pipeline
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get stage names for testing
    #[cfg(test)]
    pub fn stage_names(&self) -> Vec<&'static str> {
        self.stages.iter().map(|s| s.name()).collect()
    }

    /// Create a new pipeline with all stages
    pub fn new(worker_urls: Vec<String>) -> Self {
        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(ValidationStage),
            Box::new(ModelDiscoveryStage::new(worker_urls)),
            Box::new(ContextLoadingStage),
            Box::new(RequestBuildingStage),
            Box::new(McpPreparationStage),
            Box::new(RequestExecutionStage),
            Box::new(ResponseProcessingStage),
            Box::new(PersistenceStage),
        ];

        Self {
            stages: Arc::new(stages),
        }
    }

    /// Execute the complete pipeline for a request
    pub async fn execute(
        &self,
        request_type: RequestType,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        let mut ctx = RequestContext {
            input: RequestInput {
                request_type,
                headers,
                model_id,
            },
            components,
            state: ProcessingState::default(),
        };

        // Execute all stages sequentially
        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage completed successfully with a response (e.g., streaming)
                    return response;
                }
                Ok(None) => {
                    // Continue to next stage
                    continue;
                }
                Err(response) => {
                    // Error occurred
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        // Return final response
        match ctx.state.final_response {
            Some(final_resp) => (StatusCode::OK, Json(final_resp.json_response)).into_response(),
            None => {
                error!(function = "execute", "No response produced by pipeline");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "message": "No response produced by pipeline",
                            "type": "internal_error"
                        }
                    })),
                )
                    .into_response()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let pipeline = RequestPipeline::new(vec!["http://localhost:8000".to_string()]);
        assert_eq!(pipeline.stages.len(), 8);
    }

    #[tokio::test]
    async fn test_pipeline_stage_names() {
        let pipeline = RequestPipeline::new(vec!["http://localhost:8000".to_string()]);

        let expected_names = [
            "Validation",
            "ModelDiscovery",
            "ContextLoading",
            "RequestBuilding",
            "McpPreparation",
            "RequestExecution",
            "ResponseProcessing",
            "Persistence",
        ];

        for (idx, stage) in pipeline.stages.iter().enumerate() {
            assert_eq!(stage.name(), expected_names[idx]);
        }
    }
}
