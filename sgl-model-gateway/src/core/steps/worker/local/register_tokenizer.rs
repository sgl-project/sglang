//! Connection mode detection step.

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use tracing::{debug, warn};

use crate::{
    app_context::AppContext,
    core::Worker,
    tokenizer::{factory, TokenizerRegistry},
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Step 6: Register tokenizer for the worker's model (optional, non-blocking)
pub struct RegisterTokenizerStep;

#[async_trait]
impl StepExecutor for RegisterTokenizerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let labels: Arc<HashMap<String, String>> = context
            .get("labels")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("labels".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let workers: Arc<Vec<Arc<dyn Worker>>> = context
            .get("workers")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        for worker in workers.iter() {
            let model_id = worker.model_id().to_string();
            // Get tokenizer path (prefer tokenizer_path, fallback to model_path)
            let Some(tokenizer_path) = labels
                .get("tokenizer_path")
                .or_else(|| labels.get("model_path"))
            else {
                warn!(
                    "No tokenizer_path or model_path found for model {}",
                    model_id
                );
                return Ok(StepResult::Success);
            };

            debug!(
                "Registering tokenizer for model {} from {}",
                model_id, tokenizer_path
            );

            // Generate ID for this tokenizer
            let tokenizer_id = TokenizerRegistry::generate_id();
            let source = tokenizer_path.clone();

            // Load tokenizer with thread safe lock
            if let Err(e) = app_context
                .tokenizer_registry
                .load(&tokenizer_id, &model_id, &source, || async move {
                    factory::create_tokenizer_async(&tokenizer_path.to_string())
                        .await
                        .map_err(|e| e.to_string())
                })
                .await
            {
                warn!(
                    "Failed to load tokenizer for model {} from {}: {}",
                    model_id, source, e
                );
            } else {
                debug!(
                    "Successfully registered tokenizer for model {} from {}",
                    model_id, source
                );
            }
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // Tokenizer loading failures are retryable (network/IO issues)
    }
}
