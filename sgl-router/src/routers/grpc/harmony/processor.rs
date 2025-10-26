//! Harmony response processor for non-streaming responses

use std::sync::Arc;

use axum::response::Response;

use super::HarmonyParserAdapter;
use crate::{
    protocols::chat::ChatCompletionRequest,
    routers::grpc::{context::ExecutionResult, utils},
};

/// Processor for non-streaming Harmony responses
///
/// Collects all output tokens from execution and parses them using
/// HarmonyParserAdapter to extract the complete response.
pub struct HarmonyResponseProcessor;

impl HarmonyResponseProcessor {
    /// Create a new Harmony response processor
    pub fn new() -> Self {
        Self
    }

    /// Process a non-streaming Harmony response
    ///
    /// Takes the execution result containing output tokens and parses them
    /// into a complete ChatCompletionResponse.
    pub async fn process_harmony_response(
        &self,
        _execution_result: ExecutionResult,
        _chat_request: Arc<ChatCompletionRequest>,
        request_id: String,
    ) -> Result<Response, Response> {
        // TODO: Extract output tokens from _execution_result
        // The ExecutionResult should contain the output token IDs from the model

        // TODO: Create parser for Harmony channel parsing and use it
        // For now, we'll return a placeholder response
        let _parser = HarmonyParserAdapter::new()
            .map_err(|e| utils::bad_request_error(format!("Failed to create parser: {}", e)))?;

        // Parse the output tokens
        // For now, return a placeholder response
        // This will be fully implemented when we understand the ExecutionResult structure
        Ok(Response::builder()
            .status(200)
            .header("content-type", "application/json")
            .body(
                format!(
                    r#"{{"id":"{}","object":"chat.completion","created":{},"model":"harmony","choices":[{{"index":0,"message":{{"role":"assistant","content":"[harmony response placeholder]"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}}}"#,
                    request_id,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                )
                .into(),
            )
            .unwrap())
    }
}

impl Default for HarmonyResponseProcessor {
    fn default() -> Self {
        Self::new()
    }
}
