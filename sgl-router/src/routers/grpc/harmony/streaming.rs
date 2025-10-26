//! Harmony streaming response processor

use std::sync::Arc;

use axum::response::{sse::KeepAlive, IntoResponse, Sse};
use futures::stream;

use crate::{protocols::chat::ChatCompletionRequest, routers::grpc::context::ExecutionResult};

/// Processor for streaming Harmony responses
///
/// Returns an SSE stream that parses Harmony tokens incrementally and
/// emits ChatCompletionChunk events for streaming responses.
pub struct HarmonyStreamingProcessor;

impl HarmonyStreamingProcessor {
    /// Create a new Harmony streaming processor
    pub fn new() -> Self {
        Self
    }

    /// Process a streaming Harmony response
    ///
    /// Returns an SSE response with streaming token updates.
    pub fn process_streaming_response(
        self: Arc<Self>,
        _execution_result: ExecutionResult,
        _chat_request: Arc<ChatCompletionRequest>,
    ) -> axum::response::Response {
        // Create SSE stream for incremental token processing
        let stream = Box::pin(stream::iter(
            vec![
                // TODO: Replace with actual streaming token deltas from execution_result
                // For now, return placeholder chunks
                serde_json::json!({
                    "id": "chatcmpl-placeholder",
                    "object": "chat.completion.chunk",
                    "created": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    "model": "harmony",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "[harmony streaming placeholder]"
                        },
                        "finish_reason": null
                    }]
                }),
                serde_json::json!({
                    "id": "chatcmpl-placeholder",
                    "object": "chat.completion.chunk",
                    "created": std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    "model": "harmony",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }),
            ]
            .into_iter()
            .map(
                |json| -> Result<axum::response::sse::Event, serde_json::Error> {
                    Ok(axum::response::sse::Event::default().data(json.to_string()))
                },
            ),
        ));

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    }
}

impl Default for HarmonyStreamingProcessor {
    fn default() -> Self {
        Self::new()
    }
}
