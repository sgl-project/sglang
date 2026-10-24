//! Shared response formatting logic
//!
//! This module contains common logic for formatting responses, including:
//! - Usage calculation from gRPC responses
//! - ChatCompletionResponse construction

use crate::{
    protocols::common::Usage,
    routers::grpc::{
        proto_wrapper::ProtoGenerateComplete, utils::clamp_worker_token_count,
        utils::saturate_to_usage_u32,
    },
};

/// Build usage information from collected gRPC responses
///
/// Sums prompt_tokens and completion_tokens across all responses.
/// Typically used with n>1 parameter where multiple completions are generated.
///
/// Worker-reported counts are proto `int32` values that the gateway trusts for
/// accounting. They are clamped at zero (negative sentinels must not surface as
/// ~4.29e9 after an `as u32` cast) and summed in `u64` so the aggregate cannot
/// wrap; the result is saturated into the schema's `u32` fields.
///
/// # Arguments
/// * `responses` - Vector of GenerateComplete responses from the backend
///
/// # Returns
/// Usage object with aggregated token counts
pub(crate) fn build_usage(responses: &[ProtoGenerateComplete]) -> Usage {
    let total_prompt_tokens: u64 = responses
        .iter()
        .map(|r| clamp_worker_token_count(r.prompt_tokens()))
        .sum();
    let total_completion_tokens: u64 = responses
        .iter()
        .map(|r| clamp_worker_token_count(r.completion_tokens()))
        .sum();

    Usage {
        prompt_tokens: saturate_to_usage_u32(total_prompt_tokens),
        completion_tokens: saturate_to_usage_u32(total_completion_tokens),
        total_tokens: saturate_to_usage_u32(total_prompt_tokens + total_completion_tokens),
        completion_tokens_details: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete(prompt_tokens: i32, completion_tokens: i32) -> ProtoGenerateComplete {
        let mut msg = smg_grpc_client::sglang_proto::GenerateComplete::default();
        msg.prompt_tokens = prompt_tokens;
        msg.completion_tokens = completion_tokens;
        ProtoGenerateComplete::Sglang(msg)
    }

    #[test]
    fn sums_exact_when_within_u32_range() {
        let responses = [complete(10, 20), complete(5, 7)];
        let usage = build_usage(&responses);
        assert_eq!(usage.prompt_tokens, 15);
        assert_eq!(usage.completion_tokens, 27);
        assert_eq!(usage.total_tokens, 42);
    }

    #[test]
    fn n_choices_summing_over_u32_max_saturates_instead_of_wrapping() {
        // 5 completions x prompt_tokens = 1_859_468_194 -> true sum 9_297_340_970
        // (= 0x2_2A2A2A2A), which used to wrap to 707_406_378.
        let responses: Vec<ProtoGenerateComplete> =
            (0..5).map(|_| complete(1_859_468_194, 0)).collect();
        let usage = build_usage(&responses);
        assert_eq!(usage.prompt_tokens, u32::MAX);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, u32::MAX);
    }

    #[test]
    fn completion_sum_saturates_instead_of_wrapping() {
        let responses: Vec<ProtoGenerateComplete> =
            (0..3).map(|_| complete(0, 2_000_000_000)).collect();
        let usage = build_usage(&responses);
        assert_eq!(usage.completion_tokens, u32::MAX);
        assert_eq!(usage.total_tokens, u32::MAX);
    }

    #[test]
    fn negative_worker_reports_are_clamped_to_zero() {
        // A worker reporting prompt_tokens = -1 used to surface as
        // prompt_tokens = 4_294_967_295 and produce total < completion.
        let responses = [complete(-1, 67)];
        let usage = build_usage(&responses);
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 67);
        assert_eq!(usage.total_tokens, 67);
    }

    #[test]
    fn negative_completion_report_does_not_inflate_total() {
        let responses = [complete(100, -2_000_000_000), complete(100, 50)];
        let usage = build_usage(&responses);
        assert_eq!(usage.prompt_tokens, 200);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 250);
    }
}
