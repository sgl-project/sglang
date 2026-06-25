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
use serde_json::Value;
use tracing::{debug, error, warn};

use crate::{
    app_context::AppContext,
    core::{steps::TokenizerConfigRequest, Job, UNKNOWN_MODEL_ID},
    protocols::{
        chat::ChatCompletionRequest,
        tokenize::{
            AddTokenizerRequest, AddTokenizerResponse, CountResult, DetokenizeRequest,
            DetokenizeResponse, ListTokenizersResponse, RemoveTokenizerResponse, TextResult,
            TokenizeRequest, TokenizeResponse, TokenizerInfo, TokensResult,
        },
    },
    routers::grpc::utils::process_chat_messages,
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

    // Try UNKNOWN_MODEL_ID if model is "unknown" or empty
    if model == UNKNOWN_MODEL_ID || model.is_empty() {
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

/// Handle POST /v1/tokenize.
///
/// Accepts exactly one of `prompt` (text/batch) or `messages` (chat-style). The
/// raw body is deserialized into the upstream `openai-protocol` `TokenizeRequest`
/// / `ChatCompletionRequest` (discriminated on which key is present, since
/// `TokenizeRequest.prompt` is required and it has no `messages` field), so the
/// gateway tracks the upstream schemas without a hand-maintained mirror.
/// `messages` is rendered locally via the gRPC router's `process_chat_messages`.
pub async fn tokenize(registry: &Arc<TokenizerRegistry>, body: Value) -> Response {
    // Match the Python validator: exactly one of `prompt` / `messages`.
    let has_prompt = body.get("prompt").is_some_and(|v| !v.is_null());
    let has_messages = body.get("messages").is_some_and(|v| !v.is_null());
    if has_prompt == has_messages {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Exactly one of 'prompt' or 'messages' must be provided.",
            "invalid_request",
        );
    }

    // Deserialize into the matching upstream type, look up the tokenizer, and
    // resolve the text(s) to encode. `messages` renders the chat template once
    // (a conversation is a single, non-batch result); `prompt` may be a batch.
    // Both feed the shared encode loop below.
    let (tokenizer, texts, is_batch): (Arc<dyn Tokenizer>, Vec<String>, bool) = if has_messages {
        let chat: ChatCompletionRequest = match serde_json::from_value(body) {
            Ok(c) => c,
            Err(e) => {
                return error_response(StatusCode::BAD_REQUEST, &e.to_string(), "invalid_request")
            }
        };
        let tokenizer = match get_tokenizer(registry, &chat.model) {
            Ok(t) => t,
            Err(e) => return error_response(StatusCode::BAD_REQUEST, &e, "tokenizer_not_found"),
        };
        match process_chat_messages(&chat, tokenizer.as_ref()) {
            Ok(p) => (tokenizer, vec![p.text], false),
            Err(e) => return error_response(StatusCode::BAD_REQUEST, &e, "chat_template_error"),
        }
    } else {
        let req: TokenizeRequest = match serde_json::from_value(body) {
            Ok(r) => r,
            Err(e) => {
                return error_response(StatusCode::BAD_REQUEST, &e.to_string(), "invalid_request")
            }
        };
        let tokenizer = match get_tokenizer(registry, &req.model) {
            Ok(t) => t,
            Err(e) => return error_response(StatusCode::BAD_REQUEST, &e, "tokenizer_not_found"),
        };
        let is_batch = req.prompt.is_batch();
        (
            tokenizer,
            req.prompt
                .as_strings()
                .into_iter()
                .map(str::to_owned)
                .collect(),
            is_batch,
        )
    };

    // Encode each text without special tokens: the chat template emits BOS/EOS
    // as text, and the gateway's generate/chat paths also encode with `false`,
    // so `/tokenize` output matches what the engine actually consumes (the
    // tokenize API has used `false` since #16087). This differs from the Python
    // server's prompt-path default of `True`, but is intentional and pre-existing.
    let mut all_tokens: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
    let mut all_counts: Vec<i32> = Vec::with_capacity(texts.len());
    let mut all_char_counts: Vec<i32> = Vec::with_capacity(texts.len());
    for text in &texts {
        let token_ids = match tokenizer.encode(text, false) {
            Ok(enc) => enc.token_ids().to_vec(),
            Err(e) => {
                error!("Tokenization failed: {}", e);
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Tokenization failed: {}", e),
                    "tokenization_error",
                );
            }
        };
        all_counts.push(token_ids.len() as i32);
        all_char_counts.push(text.chars().count() as i32);
        all_tokens.push(token_ids);
    }

    // Single (non-batch) result collapses the one-element vectors. `char_count`
    // reflects the encoded text(s); the Python server returns `max_model_len`.
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
    // Note: API-initiated tokenizer loads don't use caching by default
    // Caching is applied for startup and worker-initiated loads based on router config
    let config = TokenizerConfigRequest {
        id: tokenizer_id.clone(),
        name: request.name.clone(),
        source: request.source.clone(),
        chat_template_path: request.chat_template_path.clone(),
        cache_config: None,
        fail_on_duplicate: true,
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
    use serde_json::json;

    use super::*;

    fn empty_registry() -> Arc<TokenizerRegistry> {
        Arc::new(TokenizerRegistry::new())
    }

    /// Registry holding a single (non-HF) MockTokenizer registered under `name`.
    async fn registry_with_mock(name: &str) -> Arc<TokenizerRegistry> {
        let registry = Arc::new(TokenizerRegistry::new());
        registry
            .load("mock-id", name, "mock", || async {
                Ok(Arc::new(crate::tokenizer::MockTokenizer::new()) as Arc<dyn Tokenizer>)
            })
            .await
            .expect("register mock tokenizer");
        registry
    }

    async fn read(resp: Response) -> (StatusCode, Value) {
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, json)
    }

    #[tokio::test]
    async fn rejects_both_prompt_and_messages() {
        let r = tokenize(
            &empty_registry(),
            json!({
                "model": "m",
                "prompt": "hi",
                "messages": [{"role": "user", "content": "hi"}]
            }),
        )
        .await;
        let (status, body) = read(r).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request");
    }

    #[tokio::test]
    async fn rejects_neither_prompt_nor_messages() {
        let r = tokenize(&empty_registry(), json!({ "model": "m" })).await;
        let (status, body) = read(r).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request");
    }

    #[tokio::test]
    async fn messages_with_non_hf_tokenizer_returns_400() {
        // MockTokenizer has no chat template / is not an HF tokenizer, so
        // process_chat_messages must reject it cleanly (not 500).
        let r = tokenize(
            &registry_with_mock("test-model").await,
            json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}]
            }),
        )
        .await;
        let (status, body) = read(r).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "chat_template_error");
    }

    #[tokio::test]
    async fn prompt_path_unchanged() {
        // Regression: a single text input returns a single (non-batch) result.
        let r = tokenize(
            &registry_with_mock("test-model").await,
            json!({ "model": "test-model", "prompt": "hello world" }),
        )
        .await;
        let (status, body) = read(r).await;
        assert_eq!(status, StatusCode::OK);
        // Single result: `tokens` is a flat list of ints, not a list of lists.
        assert!(
            body["tokens"][0].is_number(),
            "single (non-batch) token list"
        );
        assert!(body["count"].is_number());
    }

    #[tokio::test]
    async fn prompt_batch_returns_batch() {
        // A list `prompt` returns a batch: `tokens` is a list of token lists and
        // `count` is a list. This is the branch most sensitive to the refactor.
        let r = tokenize(
            &registry_with_mock("test-model").await,
            json!({ "model": "test-model", "prompt": ["a b", "c d e"] }),
        )
        .await;
        let (status, body) = read(r).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["tokens"].as_array().map(Vec::len), Some(2));
        assert!(body["tokens"][0].is_array(), "batch => list of token lists");
        assert!(body["count"].is_array());
    }

    #[tokio::test]
    async fn unknown_model_returns_tokenizer_not_found() {
        // Validation (exactly-one) passes, then tokenizer lookup fails -> 400.
        let r = tokenize(
            &empty_registry(),
            json!({ "model": "missing", "prompt": "hi" }),
        )
        .await;
        let (status, body) = read(r).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "tokenizer_not_found");
    }

    // The successful `messages` path requires a real HuggingFace tokenizer with a
    // chat template, so it is covered by the e2e / transformers-parity checks
    // rather than these unit tests (which use a non-HF MockTokenizer).
}
