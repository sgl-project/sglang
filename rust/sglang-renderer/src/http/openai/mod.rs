use std::sync::Arc;

use axum::Router;
use dynamo_protocols::types::CompletionUsage;

use super::OpenAIHttpFrontend;

pub(super) mod chat;
pub(super) mod completions;

pub(super) fn routes() -> Router<Arc<OpenAIHttpFrontend>> {
    Router::new()
        .merge(chat::routes())
        .merge(completions::routes())
}

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
