//! Tokenizer registration step for local workers.

use std::io::Cursor;

use async_trait::async_trait;
use tempfile::tempdir;
use tracing::{debug, info, warn};
use zip::ZipArchive;

use crate::{
    core::{steps::workflow_data::LocalWorkerWorkflowData, ConnectionMode, RuntimeType},
    tokenizer::{factory, TokenizerRegistry},
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Step 6: Register tokenizer for the worker's model (optional, non-blocking)
pub struct RegisterTokenizerStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for RegisterTokenizerStep {
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
                .load(&tokenizer_id, &model_id, &source, || {
                    let app_context = app_context.clone();
                    let source = source.clone();
                    let model_id = model_id.clone();
                    async move {
                        // 1. Try to load locally first
                        let local_result = factory::create_tokenizer_async(&source)
                            .await
                            .map_err(|e| e.to_string());
                        if local_result.is_ok() {
                            return local_result;
                        }
                        debug!(
                            "Local tokenizer load failed for source '{}', attempting to fetch from worker. Error: {:?}",
                            source,
                            local_result.err()
                        );

                        // 2. If local load fails, try to fetch from worker
                        let worker = app_context
                            .worker_registry
                            .get_workers_filtered(
                                Some(&model_id),
                                None,
                                Some(ConnectionMode::Grpc { port: None }),
                                Some(RuntimeType::Sglang),
                                true, // healthy_only
                            )
                            .into_iter()
                            .next()
                            .ok_or_else(|| {
                                "No healthy SGLang worker available to fetch tokenizer".to_string()
                            })?;

                        info!("Fetching tokenizer from worker: {}", worker.url());

                        let grpc_client = worker
                            .get_grpc_client()
                            .await
                            .map_err(|e| format!("Failed to get gRPC client: {}", e))?
                            .ok_or_else(|| "Worker does not support gRPC".to_string())?;

                        // Fetch tokenizer bundle
                        let bundle =
                            grpc_client.as_sglang().get_tokenizer().await.map_err(|e| {
                                format!("Failed to fetch tokenizer from worker: {}", e)
                            })?;

                        // Decompress to temp directory
                        let dir =
                            tempdir().map_err(|e| format!("Failed to create temp dir: {}", e))?;
                        let mut archive = ZipArchive::new(Cursor::new(bundle.compressed_data))
                            .map_err(|e| format!("Failed to open zip archive: {}", e))?;

                        archive
                            .extract(dir.path())
                            .map_err(|e| format!("Failed to extract zip archive: {}", e))?;

                        // Load from temp directory
                        let tokenizer_path = dir.path().to_str().ok_or("Invalid temp path")?;
                        info!("Tokenizer extracted to temporary path: {}", tokenizer_path);

                        factory::create_tokenizer_async(tokenizer_path)
                            .await
                            .map_err(|e| {
                                format!("Failed to load tokenizer from worker bundle: {}", e)
                            })
                    }
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
