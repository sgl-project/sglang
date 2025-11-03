//! Shared response formatting logic
//!
//! This module contains common logic for formatting responses, including:
//! - Usage calculation from gRPC responses
//! - ChatCompletionResponse construction

use crate::{
    grpc_client::proto,
    protocols::{
        chat::{ChatChoice, ChatCompletionResponse},
        common::Usage,
    },
    routers::grpc::context::DispatchMetadata,
};

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
pub fn build_usage(responses: &[proto::GenerateComplete]) -> Usage {
    let total_prompt_tokens: u32 = responses.iter().map(|r| r.prompt_tokens as u32).sum();
    let total_completion_tokens: u32 = responses.iter().map(|r| r.completion_tokens as u32).sum();

    Usage {
        prompt_tokens: total_prompt_tokens,
        completion_tokens: total_completion_tokens,
        total_tokens: total_prompt_tokens + total_completion_tokens,
        completion_tokens_details: None,
    }
}

/// Build final ChatCompletionResponse from processed choices
///
/// Constructs the OpenAI-compatible response object with all metadata.
///
/// # Arguments
/// * `choices` - Processed chat choices (after parsing, logprobs, etc.)
/// * `dispatch` - Dispatch metadata (request_id, created timestamp, etc.)
/// * `model` - Model name to include in response
/// * `usage` - Token usage information
///
/// # Returns
/// Complete ChatCompletionResponse ready to send to client
pub fn build_chat_response(
    choices: Vec<ChatChoice>,
    dispatch: &DispatchMetadata,
    model: String,
    usage: Usage,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: dispatch.request_id.clone(),
        object: "chat.completion".to_string(),
        created: dispatch.created,
        model,
        choices,
        usage: Some(usage),
        system_fingerprint: dispatch.weight_version.clone(),
    }
}
