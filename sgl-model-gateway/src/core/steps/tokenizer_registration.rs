//! Tokenizer registration workflow
//!
//! This module provides a workflow for registering tokenizers asynchronously.
//! Tokenizers can be loaded from local paths or downloaded from HuggingFace.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info};

use crate::{app_context::AppContext, tokenizer::factory, workflow::*};

/// Configuration for adding a tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfigRequest {
    /// Pre-generated UUID for this tokenizer
    pub id: String,
    /// User-provided name
    pub name: String,
    /// Source: either a local path or HuggingFace model ID
    pub source: String,
    /// Optional path to chat template file
    pub chat_template_path: Option<String>,
}

/// Configuration for removing a tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerRemovalRequest {
    /// UUID of the tokenizer to remove
    pub id: String,
}

// ============================================================================
// Workflow Steps
// ============================================================================

/// Step 1: Validate the tokenizer configuration
pub struct ValidateTokenizerConfigStep;

#[async_trait]
impl StepExecutor for ValidateTokenizerConfigStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<TokenizerConfigRequest> = context
            .get("tokenizer_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("tokenizer_config".to_string()))?;

        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        debug!(
            "Validating tokenizer config: name={}, source={}",
            config.name, config.source
        );

        // Validate name is not empty
        if config.name.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_config"),
                message: "Tokenizer name cannot be empty".to_string(),
            });
        }

        // Validate source is not empty
        if config.source.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_config"),
                message: "Tokenizer source cannot be empty".to_string(),
            });
        }

        // Check if tokenizer already exists
        if app_context.tokenizer_registry.contains(&config.name) {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_config"),
                message: format!("Tokenizer '{}' already exists", config.name),
            });
        }

        debug!("Tokenizer config validated successfully");
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Validation errors are not retryable
    }
}

/// Step 2: Load the tokenizer from source (local path or HuggingFace)
pub struct LoadTokenizerStep;

#[async_trait]
impl StepExecutor for LoadTokenizerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config: Arc<TokenizerConfigRequest> = context
            .get("tokenizer_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("tokenizer_config".to_string()))?;

        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        info!(
            "Loading tokenizer '{}' (id: {}) from source: {}",
            config.name, config.id, config.source
        );

        // Load the tokenizer using the registry's load method (handles deduplication)
        let result = app_context
            .tokenizer_registry
            .load(&config.id, &config.name, &config.source, || {
                let source = config.source.clone();
                let chat_template = config.chat_template_path.clone();
                async move {
                    factory::create_tokenizer_async_with_chat_template(
                        &source,
                        chat_template.as_deref(),
                    )
                    .await
                    .map_err(|e| format!("Failed to load tokenizer: {}", e))
                }
            })
            .await;

        match result {
            Ok(loaded_id) => {
                // Get vocab size for logging
                let vocab_size = app_context
                    .tokenizer_registry
                    .get_by_id(&loaded_id)
                    .map(|e| e.tokenizer.vocab_size());

                info!(
                    "Successfully loaded tokenizer '{}' (id: {}) with vocab_size: {:?}",
                    config.name, loaded_id, vocab_size
                );

                // Store vocab size in context for later use
                if let Some(size) = vocab_size {
                    context.set("vocab_size", size);
                }

                Ok(StepResult::Success)
            }
            Err(e) => {
                error!("Failed to load tokenizer '{}': {}", config.name, e);
                Err(WorkflowError::StepFailed {
                    step_id: StepId::new("load_tokenizer"),
                    message: e,
                })
            }
        }
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // Network/IO errors are retryable
    }
}

// ============================================================================
// Workflow Builder
// ============================================================================

/// Create the tokenizer registration workflow
///
/// This workflow:
/// - Validates the tokenizer configuration
/// - Loads the tokenizer from local path or HuggingFace
///
/// Workflow configuration:
/// - ValidateConfig: No retry, 5s timeout (fast validation)
/// - LoadTokenizer: 3 retries, 5min timeout (may need to download from HuggingFace)
pub fn create_tokenizer_registration_workflow() -> WorkflowDefinition {
    WorkflowDefinition::new("tokenizer_registration", "Tokenizer Registration")
        .add_step(
            StepDefinition::new(
                "validate_config",
                "Validate Configuration",
                Arc::new(ValidateTokenizerConfigStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "load_tokenizer",
                "Load Tokenizer",
                Arc::new(LoadTokenizerStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(2)),
            })
            .with_timeout(Duration::from_secs(300)) // 5 min for HuggingFace downloads
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["validate_config"]),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_config_request_serialization() {
        let config = TokenizerConfigRequest {
            id: "test-uuid-1234".to_string(),
            name: "test-model".to_string(),
            source: "meta-llama/Llama-2-7b-hf".to_string(),
            chat_template_path: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, "test-uuid-1234");
        assert_eq!(parsed.name, "test-model");
        assert_eq!(parsed.source, "meta-llama/Llama-2-7b-hf");
        assert!(parsed.chat_template_path.is_none());
    }

    #[test]
    fn test_workflow_creation() {
        let workflow = create_tokenizer_registration_workflow();
        assert_eq!(workflow.id.to_string(), "tokenizer_registration");
    }
}
