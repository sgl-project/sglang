//! Tokenizer registration step for local workers.
//!
//! This step submits a Job::AddTokenizer to the job queue, which triggers the
//! tokenizer_registration workflow. The workflow handles validation, deduplication,
//! and caching - this step just submits the job.

use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::{
    core::{
        steps::{workflow_data::LocalWorkerWorkflowData, TokenizerConfigRequest},
        Job,
    },
    tokenizer::TokenizerRegistry,
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Step: Submit tokenizer registration job for the worker's model
///
/// This step submits a Job::AddTokenizer to the job queue rather than loading
/// the tokenizer directly. This ensures tokenizer registration goes through
/// the unified tokenizer_registration workflow.
pub struct SubmitTokenizerJobStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for SubmitTokenizerJobStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<LocalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let labels = &context.data.final_labels;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let workers = context
            .data
            .actual_workers
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        // Get job queue
        let job_queue = match app_context.worker_job_queue.get() {
            Some(queue) => queue,
            None => {
                warn!("Job queue not available, skipping tokenizer registration");
                return Ok(StepResult::Success);
            }
        };

        // Get chat_template: worker config > global router config
        let chat_template = context
            .data
            .config
            .chat_template
            .clone()
            .or_else(|| app_context.router_config.chat_template.clone());

        // Get cache config from router config
        let cache_config = app_context.router_config.tokenizer_cache.to_option();

        for worker in workers.iter() {
            let model_id = worker.model_id().to_string();

            // Get tokenizer path with fallback chain:
            // 1. Worker labels: tokenizer_path
            // 2. Worker labels: model_path
            // 3. Router config (CLI args): --tokenizer-path
            // 4. Router config (CLI args): --model-path
            let tokenizer_path: String = if let Some(path) = labels
                .get("tokenizer_path")
                .or_else(|| labels.get("model_path"))
            {
                path.clone()
            } else if let Some(path) = app_context
                .router_config
                .tokenizer_path
                .as_ref()
                .or(app_context.router_config.model_path.as_ref())
            {
                debug!(
                    "Using router config tokenizer path '{}' for model {}",
                    path, model_id
                );
                path.clone()
            } else {
                warn!(
                    "No tokenizer_path or model_path found for model {} (checked worker labels and router config)",
                    model_id
                );
                continue;
            };

            // Note: We don't check if tokenizer already exists here.
            // The registry.load() handles deduplication gracefully (returns AlreadyExists).
            // This simplifies the code and ensures consistent behavior.

            info!(
                "Submitting tokenizer registration job for model {} from {}",
                model_id, tokenizer_path
            );

            // Create tokenizer config request
            let config = TokenizerConfigRequest {
                id: TokenizerRegistry::generate_id(),
                name: model_id.clone(),
                source: tokenizer_path,
                chat_template_path: chat_template.clone(),
                cache_config: cache_config.clone(),
                fail_on_duplicate: false,
            };

            // Submit job (fire-and-forget, don't wait for completion)
            if let Err(e) = job_queue
                .submit(Job::AddTokenizer {
                    config: Box::new(config),
                })
                .await
            {
                warn!(
                    "Failed to submit tokenizer job for model {}: {}",
                    model_id, e
                );
            }
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Job submission failures are not retryable at this level
    }
}
