//! Tokenize and detokenize handlers
//!
//! Provides tokenization, detokenization, and tokenizer management operations.
//! These handlers use the TokenizerRegistry for tokenizer storage and retrieval.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use tracing::{debug, error, warn};

use crate::{
    app_context::AppContext,
    core::{steps::TokenizerConfigRequest, Job},
    protocols::tokenize::{
        AddTokenizerRequest, AddTokenizerResponse, CountResult, DetokenizeRequest,
        DetokenizeResponse, ListTokenizersResponse, RemoveTokenizerResponse, TextResult,
        TokenizeRequest, TokenizeResponse, TokenizerInfo, TokensResult,
    },
    tokenizer::{registry::TokenizerEntry, traits::Tokenizer, TokenizerRegistry},
};

/// Helper to create error responses
fn error_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": error_type
            }
        })),
    )
        .into_response()
}

/// Get a tokenizer by model name, with fallback strategies
fn get_tokenizer(registry: &TokenizerRegistry, model: &str) -> Result<Arc<dyn Tokenizer>, String> {
    // First, try exact match (by name or ID)
    if let Some(tokenizer) = registry.get(model) {
        debug!("Found tokenizer for model: {}", model);
        return Ok(tokenizer);
    }

    // Try "default" if model is "default" or empty
    if model == "default" || model.is_empty() {
        // Try to find any tokenizer as fallback
        let entries = registry.list();
        if let Some(first) = entries.first() {
            debug!(
                "Using first available tokenizer '{}' as default",
                first.name
            );
            return Ok(first.tokenizer.clone());
        }
    }

    // List available tokenizers for error message
    let entries = registry.list();
    if entries.is_empty() {
        Err("No tokenizers available. Use POST /v1/tokenizers to add one.".to_string())
    } else {
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        Err(format!(
            "Tokenizer for model '{}' not found. Available: {}",
            model,
            names.join(", ")
        ))
    }
}

// ============================================================================
// Tokenize / Detokenize Handlers
// ============================================================================

/// Handle POST /v1/tokenize
pub async fn tokenize(registry: &Arc<TokenizerRegistry>, request: TokenizeRequest) -> Response {
    debug!("Tokenize request for model: {}", request.model);

    let tokenizer = match get_tokenizer(registry, &request.model) {
        Ok(t) => t,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, &e, "tokenizer_not_found");
        }
    };

    let texts = request.prompt.as_strings();
    let is_batch = request.prompt.is_batch();

    // Tokenize each text
    let mut all_tokens: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
    let mut all_counts: Vec<i32> = Vec::with_capacity(texts.len());
    let mut all_char_counts: Vec<i32> = Vec::with_capacity(texts.len());

    for text in texts {
        // Don't add special tokens for tokenize API (matches Python behavior)
        let encoding = match tokenizer.encode(text, false) {
            Ok(enc) => enc,
            Err(e) => {
                error!("Tokenization failed: {}", e);
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Tokenization failed: {}", e),
                    "tokenization_error",
                );
            }
        };

        let token_ids: Vec<u32> = encoding.token_ids().to_vec();
        let count = token_ids.len() as i32;

        all_tokens.push(token_ids);
        all_counts.push(count);
        all_char_counts.push(text.chars().count() as i32);
    }

    // Format response based on single vs batch
    let (tokens, count, char_count) = if is_batch {
        (
            TokensResult::Batch(all_tokens),
            CountResult::Batch(all_counts),
            CountResult::Batch(all_char_counts),
        )
    } else {
        (
            TokensResult::Single(all_tokens.into_iter().next().unwrap_or_default()),
            CountResult::Single(all_counts.into_iter().next().unwrap_or(0)),
            CountResult::Single(all_char_counts.into_iter().next().unwrap_or(0)),
        )
    };

    Json(TokenizeResponse {
        tokens,
        count,
        char_count,
    })
    .into_response()
}

/// Handle POST /v1/detokenize
pub async fn detokenize(registry: &Arc<TokenizerRegistry>, request: DetokenizeRequest) -> Response {
    debug!("Detokenize request for model: {}", request.model);

    let tokenizer = match get_tokenizer(registry, &request.model) {
        Ok(t) => t,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, &e, "tokenizer_not_found");
        }
    };

    let sequences = request.tokens.sequences();
    let is_batch = request.tokens.is_batch();

    // Detokenize each sequence
    let mut all_texts: Vec<String> = Vec::with_capacity(sequences.len());

    for seq in sequences {
        let text = match tokenizer.decode(seq, request.skip_special_tokens) {
            Ok(t) => t,
            Err(e) => {
                error!("Detokenization failed: {}", e);
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Detokenization failed: {}", e),
                    "detokenization_error",
                );
            }
        };
        all_texts.push(text);
    }

    // Format response based on single vs batch
    let text = if is_batch {
        TextResult::Batch(all_texts)
    } else {
        TextResult::Single(all_texts.into_iter().next().unwrap_or_default())
    };

    Json(DetokenizeResponse { text }).into_response()
}

// ============================================================================
// Tokenizer Management Handlers
// ============================================================================

/// Handle POST /v1/tokenizers - async version using job queue
pub async fn add_tokenizer(context: &Arc<AppContext>, request: AddTokenizerRequest) -> Response {
    // Check if tokenizer already exists by name
    if context.tokenizer_registry.contains(&request.name) {
        // Return the existing tokenizer's ID
        if let Some(entry) = context.tokenizer_registry.get_by_name(&request.name) {
            return (
                StatusCode::CONFLICT,
                Json(AddTokenizerResponse {
                    id: entry.id,
                    status: "failed".to_string(),
                    message: format!("Tokenizer '{}' already exists", request.name),
                    vocab_size: None,
                }),
            )
                .into_response();
        }
    }

    // Get the job queue
    let job_queue = match context.worker_job_queue.get() {
        Some(queue) => queue,
        None => {
            error!("Job queue not available");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(AddTokenizerResponse {
                    id: String::new(),
                    status: "failed".to_string(),
                    message: "Job queue not available".to_string(),
                    vocab_size: None,
                }),
            )
                .into_response();
        }
    };

    // Generate UUID for this tokenizer
    let tokenizer_id = TokenizerRegistry::generate_id();

    // Create the job with the pre-generated ID
    let config = TokenizerConfigRequest {
        id: tokenizer_id.clone(),
        name: request.name.clone(),
        source: request.source.clone(),
        chat_template_path: request.chat_template_path.clone(),
    };

    let job = Job::AddTokenizer {
        config: Box::new(config),
    };

    // Submit the job
    match job_queue.submit(job).await {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(AddTokenizerResponse {
                id: tokenizer_id,
                status: "pending".to_string(),
                message: format!(
                    "Tokenizer '{}' registration job submitted. Loading from: {}",
                    request.name, request.source
                ),
                vocab_size: None,
            }),
        )
            .into_response(),
        Err(e) => {
            error!("Failed to submit tokenizer job: {}", e);
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(AddTokenizerResponse {
                    id: String::new(),
                    status: "failed".to_string(),
                    message: e,
                    vocab_size: None,
                }),
            )
                .into_response()
        }
    }
}

/// Handle GET /v1/tokenizers
pub async fn list_tokenizers(registry: &Arc<TokenizerRegistry>) -> Response {
    debug!("List tokenizers request");

    let entries = registry.list();
    let tokenizers: Vec<TokenizerInfo> = entries
        .into_iter()
        .map(|e| TokenizerInfo {
            id: e.id,
            name: e.name,
            source: e.source,
            vocab_size: e.tokenizer.vocab_size(),
        })
        .collect();

    Json(ListTokenizersResponse { tokenizers }).into_response()
}

/// Handle DELETE /v1/tokenizers/{tokenizer_id}
pub async fn remove_tokenizer(context: &Arc<AppContext>, tokenizer_id: &str) -> Response {
    // Try to remove by ID first, then by name for backward compatibility
    let removed = context
        .tokenizer_registry
        .remove_by_id(tokenizer_id)
        .or_else(|| context.tokenizer_registry.remove(tokenizer_id));

    if let Some(entry) = removed {
        debug!("Removed tokenizer '{}' (id: {})", entry.name, entry.id);
        (
            StatusCode::OK,
            Json(RemoveTokenizerResponse {
                success: true,
                message: format!("Tokenizer '{}' removed successfully", entry.name),
            }),
        )
            .into_response()
    } else {
        warn!("Tokenizer '{}' not found", tokenizer_id);
        (
            StatusCode::NOT_FOUND,
            Json(RemoveTokenizerResponse {
                success: false,
                message: format!("Tokenizer '{}' not found", tokenizer_id),
            }),
        )
            .into_response()
    }
}

/// Handle GET /v1/tokenizers/{tokenizer_id}
pub async fn get_tokenizer_info(context: &Arc<AppContext>, tokenizer_id: &str) -> Response {
    debug!("Get tokenizer info for '{}'", tokenizer_id);

    // Try by ID first, then by name
    let entry: Option<TokenizerEntry> = context
        .tokenizer_registry
        .get_by_id(tokenizer_id)
        .or_else(|| context.tokenizer_registry.get_by_name(tokenizer_id));

    match entry {
        Some(e) => {
            let info = TokenizerInfo {
                id: e.id,
                name: e.name,
                source: e.source,
                vocab_size: e.tokenizer.vocab_size(),
            };
            Json(info).into_response()
        }
        None => error_response(
            StatusCode::NOT_FOUND,
            &format!("Tokenizer '{}' not found", tokenizer_id),
            "tokenizer_not_found",
        ),
    }
}

/// Handle GET /v1/tokenizers/{tokenizer_id}/status
pub async fn get_tokenizer_status(context: &Arc<AppContext>, tokenizer_id: &str) -> Response {
    debug!("Get tokenizer status for '{}'", tokenizer_id);

    // First check if tokenizer is already loaded (by ID or name)
    let entry = context
        .tokenizer_registry
        .get_by_id(tokenizer_id)
        .or_else(|| context.tokenizer_registry.get_by_name(tokenizer_id));

    if let Some(e) = entry {
        return Json(AddTokenizerResponse {
            id: e.id,
            status: "completed".to_string(),
            message: format!("Tokenizer '{}' is loaded and ready", e.name),
            vocab_size: Some(e.tokenizer.vocab_size()),
        })
        .into_response();
    }

    // Check job status (jobs are tracked by ID)
    if let Some(job_queue) = context.worker_job_queue.get() {
        if let Some(job_status) = job_queue.get_status(tokenizer_id) {
            return Json(AddTokenizerResponse {
                id: tokenizer_id.to_string(),
                status: job_status.status.clone(),
                message: job_status
                    .message
                    .unwrap_or_else(|| format!("Tokenizer job is {}", job_status.status)),
                vocab_size: None,
            })
            .into_response();
        }
    }

    // Not found
    error_response(
        StatusCode::NOT_FOUND,
        &format!("Tokenizer '{}' not found and no pending job", tokenizer_id),
        "not_found",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::mock::MockTokenizer;

    fn create_test_registry() -> Arc<TokenizerRegistry> {
        let registry = Arc::new(TokenizerRegistry::new());
        let id = TokenizerRegistry::generate_id();
        registry.register(
            &id,
            "test-model",
            "test-source",
            Arc::new(MockTokenizer::new()),
        );
        registry
    }

    #[test]
    fn test_get_tokenizer_exact_match() {
        let registry = create_test_registry();
        let result = get_tokenizer(&registry, "test-model");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_tokenizer_default_fallback() {
        let registry = create_test_registry();
        let result = get_tokenizer(&registry, "default");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_tokenizer_not_found() {
        let registry = create_test_registry();
        let result = get_tokenizer(&registry, "nonexistent");
        match result {
            Err(e) => assert!(e.contains("not found")),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_get_tokenizer_empty_registry() {
        let registry = Arc::new(TokenizerRegistry::new());
        let result = get_tokenizer(&registry, "any");
        match result {
            Err(e) => assert!(e.contains("No tokenizers available")),
            Ok(_) => panic!("Expected error"),
        }
    }
}
