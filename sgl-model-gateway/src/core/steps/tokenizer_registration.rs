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
use tracing::{error, info};

use super::workflow_data::TokenizerWorkflowData;
use crate::{
    app_context::AppContext,
    config::TokenizerCacheConfig,
    tokenizer::{
        cache::{CacheConfig, CachedTokenizer},
        factory,
        registry::LoadOutcome,
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
    /// If true, the workflow fails when a tokenizer with the same name already exists.
    /// If false (default), the workflow succeeds and returns the existing tokenizer's ID.
    /// API callers should set this to true.
    #[serde(default)]
    pub fail_on_duplicate: bool,
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

/// Load the tokenizer from source (local path or HuggingFace)
///
/// This step handles:
/// - Input validation (via registry.load())
/// - Deduplication (returns success if already exists)
/// - Loading from local path or HuggingFace
/// - Optional caching layer wrapping
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

        // Load the tokenizer using the registry's load method
        // This handles: validation, deduplication, and loading
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
            Ok(outcome) => {
                let loaded_id = outcome.id();

                // Get vocab size for logging
                let vocab_size = app_context
                    .tokenizer_registry
                    .get_by_id(loaded_id)
                    .map(|e| e.tokenizer.vocab_size());

                match &outcome {
                    LoadOutcome::Loaded { id } => {
                        info!(
                            "Successfully loaded tokenizer '{}' (id: {}) with vocab_size: {:?}",
                            name, id, vocab_size
                        );
                    }
                    LoadOutcome::AlreadyExists { id } => {
                        if config.fail_on_duplicate {
                            return Err(WorkflowError::StepFailed {
                                step_id: StepId::new("load_tokenizer"),
                                message: format!(
                                    "Tokenizer '{}' already exists (id: {})",
                                    name, id
                                ),
                            });
                        }
                        info!(
                            "Tokenizer '{}' already exists (id: {}), skipping load",
                            name, id
                        );
                    }
                }

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
                    message: e.to_string(),
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
/// This workflow loads and registers a tokenizer. The single LoadTokenizerStep handles:
/// - Input validation (empty name/source)
/// - Deduplication (returns success if already exists)
/// - Loading from local path or HuggingFace
/// - Optional caching layer wrapping
///
/// Configuration:
/// - 3 retries with 2s backoff (for network issues)
/// - 5 minute timeout (HuggingFace downloads can be slow)
pub fn create_tokenizer_registration_workflow() -> WorkflowDefinition<TokenizerWorkflowData> {
    WorkflowDefinition::new("tokenizer_registration", "Tokenizer Registration").add_step(
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
        .with_failure_action(FailureAction::FailWorkflow),
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
            fail_on_duplicate: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, "test-uuid-1234");
        assert_eq!(parsed.name, "test-model");
        assert_eq!(parsed.source, "meta-llama/Llama-2-7b-hf");
        assert!(parsed.chat_template_path.is_none());
        assert!(parsed.cache_config.is_none());
        assert!(!parsed.fail_on_duplicate);
    }

    #[test]
    fn test_tokenizer_config_request_fail_on_duplicate_defaults_to_false() {
        // Test that fail_on_duplicate defaults to false when not specified in JSON
        let json = r#"{
            "id": "test-uuid",
            "name": "test-model",
            "source": "/path/to/tokenizer"
        }"#;
        let parsed: TokenizerConfigRequest = serde_json::from_str(json).unwrap();
        assert!(!parsed.fail_on_duplicate);
    }

    #[test]
    fn test_tokenizer_config_request_fail_on_duplicate_true() {
        let config = TokenizerConfigRequest {
            id: "test-uuid-1234".to_string(),
            name: "test-model".to_string(),
            source: "meta-llama/Llama-2-7b-hf".to_string(),
            chat_template_path: None,
            cache_config: None,
            fail_on_duplicate: true,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();
        assert!(parsed.fail_on_duplicate);
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
            fail_on_duplicate: false,
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
