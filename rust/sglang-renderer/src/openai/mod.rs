//! OpenAI request preparation and typed response construction.

use crate::{RendererError, RendererErrorKind, ResponseError};
use dynamo_protocols::types::CompletionUsage;

pub(crate) mod chat;
pub(crate) mod completions;
pub(crate) mod protocol;
pub(crate) mod render;
pub(crate) mod submission;
pub(crate) mod tokenize;

#[cfg(test)]
pub(crate) mod test_utils;
#[cfg(test)]
mod tests;

pub(crate) fn renderer_error(error: RendererError) -> ResponseError {
    let status_code = match error.kind() {
        RendererErrorKind::InvalidRequest => 400,
        RendererErrorKind::Unavailable => 503,
        RendererErrorKind::Tokenize | RendererErrorKind::Internal => 500,
    };
    ResponseError {
        status_code,
        message: error.to_string(),
    }
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
