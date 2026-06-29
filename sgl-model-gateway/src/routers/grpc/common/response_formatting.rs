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
pub(crate) fn build_usage(responses: &[ProtoGenerateComplete]) -> Usage {
    let total_prompt_tokens: u32 = responses.iter().map(|r| r.prompt_tokens() as u32).sum();
    let total_completion_tokens: u32 = responses.iter().map(|r| r.completion_tokens() as u32).sum();
    let total_cached_tokens: u32 = responses.iter().map(|r| r.cached_tokens() as u32).sum();

    Usage::from_counts(total_prompt_tokens, total_completion_tokens)
        .with_cached_tokens(total_cached_tokens)
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::sglang_proto;

    use super::*;

    fn complete(
        prompt_tokens: i32,
        completion_tokens: i32,
        cached_tokens: i32,
    ) -> ProtoGenerateComplete {
        ProtoGenerateComplete::Sglang(sglang_proto::GenerateComplete {
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            ..Default::default()
        })
    }

    #[test]
    fn build_usage_includes_non_zero_cached_tokens() {
        let usage = build_usage(&[complete(100, 20, 42)]);

        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 120);
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens),
            Some(42)
        );
        assert!(usage.completion_tokens_details.is_none());
    }

    #[test]
    fn build_usage_omits_zero_cached_tokens() {
        let usage = build_usage(&[complete(100, 20, 0)]);

        assert!(usage.prompt_tokens_details.is_none());
        assert!(usage.completion_tokens_details.is_none());
    }

    #[test]
    fn build_usage_sums_cached_tokens_across_responses() {
        let usage = build_usage(&[complete(100, 20, 10), complete(50, 5, 20)]);

        assert_eq!(usage.prompt_tokens, 150);
        assert_eq!(usage.completion_tokens, 25);
        assert_eq!(usage.total_tokens, 175);
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens),
            Some(30)
        );
    }
}
