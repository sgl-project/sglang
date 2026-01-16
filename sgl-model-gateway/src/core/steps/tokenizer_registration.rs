//! Tokenizer registration workflow
//!
//! This module provides a workflow for registering tokenizers asynchronously.
//! Tokenizers can be loaded from local paths or downloaded from HuggingFace.
//!
//! This is the **single source of truth** for tokenizer registration. All paths
//! (startup, worker connection, API) should use this workflow to ensure consistent
//! behavior (validation, caching, deduplication).

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info};

use super::workflow_data::TokenizerWorkflowData;
use crate::{
    app_context::AppContext,
    config::TokenizerCacheConfig,
    tokenizer::{
        cache::{CacheConfig, CachedTokenizer},
        factory,
        traits::Tokenizer,
    },
    workflow::{
        BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, StepExecutor, StepId,
        StepResult, WorkflowContext, WorkflowDefinition, WorkflowError, WorkflowResult,
    },
};

/// Configuration for adding a tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfigRequest {
    /// Pre-generated UUID for this tokenizer
    pub id: String,
    /// User-provided name (what to register under in the registry)
    pub name: String,
    /// Source: either a local path or HuggingFace model ID
    pub source: String,
    /// Optional path to chat template file
    pub chat_template_path: Option<String>,
    /// Optional cache configuration. If provided, wraps tokenizer with CachedTokenizer.
    #[serde(default)]
    pub cache_config: Option<TokenizerCacheConfig>,
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
impl StepExecutor<TokenizerWorkflowData> for ValidateTokenizerConfigStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<TokenizerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
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
impl StepExecutor<TokenizerWorkflowData> for LoadTokenizerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<TokenizerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();

        info!(
            "Loading tokenizer '{}' (id: {}) from source: {}{}",
            config.name,
            config.id,
            config.source,
            if config.cache_config.is_some() {
                " with caching"
            } else {
                ""
            }
        );

        // Clone needed values before async move
        let id = config.id.clone();
        let name = config.name.clone();
        let source = config.source.clone();
        let chat_template = config.chat_template_path.clone();
        let cache_config = config.cache_config.clone();

        // Load the tokenizer using the registry's load method (handles deduplication)
        let result = app_context
            .tokenizer_registry
            .load(&id, &name, &source, || {
                let source = source.clone();
                let chat_template = chat_template.clone();
                let cache_cfg = cache_config.clone();
                async move {
                    // Load base tokenizer
                    let base_tokenizer = factory::create_tokenizer_async_with_chat_template(
                        &source,
                        chat_template.as_deref(),
                    )
                    .await
                    .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

                    // Wrap with caching layer if configured
                    let tokenizer: Arc<dyn Tokenizer> = match cache_cfg {
                        Some(cfg) if cfg.enable_l0 || cfg.enable_l1 => {
                            let cache_config = CacheConfig {
                                enable_l0: cfg.enable_l0,
                                l0_max_entries: cfg.l0_max_entries,
                                enable_l1: cfg.enable_l1,
                                l1_max_memory: cfg.l1_max_memory,
                            };
                            Arc::new(CachedTokenizer::new(base_tokenizer, cache_config))
                        }
                        _ => base_tokenizer,
                    };

                    Ok(tokenizer)
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
                    name, loaded_id, vocab_size
                );

                // Store vocab size in typed data
                if let Some(size) = vocab_size {
                    context.data.vocab_size = Some(size);
                }

                Ok(StepResult::Success)
            }
            Err(e) => {
                error!("Failed to load tokenizer '{}': {}", name, e);
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
pub fn create_tokenizer_registration_workflow() -> WorkflowDefinition<TokenizerWorkflowData> {
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

/// Helper to create initial workflow data for tokenizer registration
pub fn create_tokenizer_workflow_data(
    config: TokenizerConfigRequest,
    app_context: Arc<AppContext>,
) -> TokenizerWorkflowData {
    TokenizerWorkflowData {
        config,
        vocab_size: None,
        app_context: Some(app_context),
    }
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
            cache_config: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, "test-uuid-1234");
        assert_eq!(parsed.name, "test-model");
        assert_eq!(parsed.source, "meta-llama/Llama-2-7b-hf");
        assert!(parsed.chat_template_path.is_none());
        assert!(parsed.cache_config.is_none());
    }

    #[test]
    fn test_tokenizer_config_request_with_cache() {
        let config = TokenizerConfigRequest {
            id: "test-uuid-1234".to_string(),
            name: "test-model".to_string(),
            source: "meta-llama/Llama-2-7b-hf".to_string(),
            chat_template_path: None,
            cache_config: Some(TokenizerCacheConfig {
                enable_l0: true,
                l0_max_entries: 1000,
                enable_l1: false,
                l1_max_memory: 0,
            }),
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();

        assert!(parsed.cache_config.is_some());
        let cache = parsed.cache_config.unwrap();
        assert!(cache.enable_l0);
        assert_eq!(cache.l0_max_entries, 1000);
        assert!(!cache.enable_l1);
    }

    #[test]
    fn test_workflow_creation() {
        let mut workflow = create_tokenizer_registration_workflow();
        assert_eq!(workflow.id.to_string(), "tokenizer_registration");
        // Validate the workflow DAG
        workflow
            .validate()
            .expect("Workflow validation should pass");
    }
}
