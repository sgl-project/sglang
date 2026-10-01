//! OpenAI request preparation and typed response construction.

use crate::ResponseError;
use dynamo_protocols::types::CompletionUsage;

pub(crate) mod chat;
pub(crate) mod completions;
pub(crate) mod protocol;
pub(crate) mod render;
pub(crate) mod tokenize;

#[cfg(test)]
pub(crate) mod test_utils;
#[cfg(test)]
mod tests;

pub(super) fn unix_seconds_u32() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| u32::try_from(duration.as_secs()).unwrap_or(u32::MAX))
        .unwrap_or(0)
}

pub(super) fn completion_usage(prompt_tokens: u32, completion_tokens: u32) -> CompletionUsage {
    CompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
        ..Default::default()
    }
}

/// Typed route result; transport adapters supply framing and status policy.
pub(crate) enum OperationResponse<U, C> {
    Unary(U),
    Stream(futures::stream::BoxStream<'static, Result<C, ResponseError>>),
}

pub(crate) struct OpenAIService {
    pub(crate) renderer: std::sync::Arc<crate::RendererService>,
    generation: crate::engine::GenerationService,
}

impl OpenAIService {
    pub(crate) fn new(
        renderer: std::sync::Arc<crate::RendererService>,
        generation: crate::engine::GenerationService,
    ) -> Self {
        Self {
            renderer,
            generation,
        }
    }
}

pub(crate) fn error_payload(
    code: u16,
    message: impl Into<String>,
    error_type: &str,
) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "object": "error", "message": message.into(), "type": error_type,
            "param": null, "code": code,
        }
    })
}
