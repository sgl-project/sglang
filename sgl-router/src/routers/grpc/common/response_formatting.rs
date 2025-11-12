//! Shared response formatting logic
//!
//! This module contains common logic for formatting responses, including:
//! - Usage calculation from gRPC responses
//! - ChatCompletionResponse construction

use crate::{protocols::common::Usage, routers::grpc::proto_wrapper::ProtoGenerateComplete};

/// Build usage information from collected gRPC responses
///
/// Sums prompt_tokens and completion_tokens across all responses.
/// Typically used with n>1 parameter where multiple completions are generated.
///
/// # Arguments
/// * `responses` - Vector of GenerateComplete responses from the backend
///
/// # Returns
/// Usage object with aggregated token counts
pub fn build_usage(responses: &[ProtoGenerateComplete]) -> Usage {
    let total_prompt_tokens: u32 = responses.iter().map(|r| r.prompt_tokens() as u32).sum();
    let total_completion_tokens: u32 = responses.iter().map(|r| r.completion_tokens() as u32).sum();

    Usage {
        prompt_tokens: total_prompt_tokens,
        completion_tokens: total_completion_tokens,
        total_tokens: total_prompt_tokens + total_completion_tokens,
        completion_tokens_details: None,
    }
}
