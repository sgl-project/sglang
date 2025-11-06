//! Shared response collection logic
//!
//! This module contains common logic for collecting responses from execution results.
//! Both regular and harmony processors use these functions to avoid duplication.

use axum::response::Response;

use crate::{
    grpc_client::proto,
    routers::grpc::{context::ExecutionResult, error, utils},
};

/// Collect and merge responses from execution result
///
/// Handles both Single and Dual (prefill-decode) execution modes.
/// For Dual mode, merges prefill input_logprobs into decode responses if requested.
///
/// # Arguments
/// * `execution_result` - The execution result containing stream(s)
/// * `merge_logprobs` - Whether to merge prefill input_logprobs (for chat with logprobs=true)
///
/// # Returns
/// Vector of GenerateComplete responses, one per index (n parameter)
pub async fn collect_responses(
    execution_result: ExecutionResult,
    merge_logprobs: bool,
) -> Result<Vec<proto::GenerateComplete>, Response> {
    let all_responses = match execution_result {
        ExecutionResult::Single { mut stream } => {
            let responses = utils::collect_stream_responses(&mut stream, "Single").await?;
            stream.mark_completed();
            responses
        }
        ExecutionResult::Dual {
            mut prefill,
            decode,
        } => {
            // Collect prefill for input_logprobs (don't mark completed yet)
            let prefill_responses =
                utils::collect_stream_responses(&mut prefill, "Prefill").await?;

            // Collect decode for actual output (don't mark completed yet)
            let mut decode_stream = *decode;
            let mut decode_responses =
                utils::collect_stream_responses(&mut decode_stream, "Decode").await?;

            // Mark both streams as completed now that both succeeded
            prefill.mark_completed();
            decode_stream.mark_completed();

            // Merge prefill input_logprobs if requested
            if merge_logprobs {
                merge_prefill_logprobs(&prefill_responses, &mut decode_responses);
            }

            decode_responses
        }
    };

    if all_responses.is_empty() {
        return Err(error::internal_error("No responses from server"));
    }

    Ok(all_responses)
}

/// Merge prefill input_logprobs into decode responses
///
/// Takes input_logprobs from the first prefill response and copies them
/// into all decode responses. This is used in PD mode when logprobs are requested.
fn merge_prefill_logprobs(
    prefill_responses: &[proto::GenerateComplete],
    decode_responses: &mut [proto::GenerateComplete],
) {
    if let Some(prefill_input_logprobs) = prefill_responses
        .first()
        .and_then(|r| r.input_logprobs.clone())
    {
        for response in decode_responses.iter_mut() {
            response.input_logprobs = Some(prefill_input_logprobs.clone());
        }
    }
}
