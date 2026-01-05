//! Shared response collection logic
//!
//! This module contains common logic for collecting responses from execution results.
//! Both regular and harmony processors use these functions to avoid duplication.

use axum::response::Response;

use crate::routers::{
    error,
    grpc::{
        context::ExecutionResult,
        proto_wrapper::{ProtoGenerateComplete, ProtoStream},
        utils,
    },
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
pub(crate) async fn collect_responses(
    execution_result: ExecutionResult,
    merge_logprobs: bool,
) -> Result<Vec<ProtoGenerateComplete>, Response> {
    let all_responses = match execution_result {
        ExecutionResult::Single { mut stream } => {
            let responses = utils::collect_stream_responses(&mut stream, "Single").await?;
            stream.mark_completed();
            responses
        }
        ExecutionResult::Dual {
            mut prefill,
            decode,
        } => collect_prefill_decode(&mut prefill, decode, merge_logprobs).await?,
        ExecutionResult::Embedding { .. } => {
            // Embeddings do not support this path (no generate complete response)
            return Err(error::internal_error(
                "invalid_execution_mode",
                "Embedding result encountered in response collection",
            ));
        }
    };

    if all_responses.is_empty() {
        return Err(error::internal_error(
            "no_responses_from_server",
            "No responses from server",
        ));
    }

    Ok(all_responses)
}

/// Collect prefill and decode streams, merge input_logprobs if requested
///
/// Common logic for Dual (PD) mode.
async fn collect_prefill_decode(
    prefill: &mut ProtoStream,
    decode: Box<ProtoStream>,
    merge_logprobs: bool,
) -> Result<Vec<ProtoGenerateComplete>, Response> {
    let prefill_responses = utils::collect_stream_responses(prefill, "Prefill").await?;

    let mut decode_stream = *decode;
    let mut decode_responses =
        utils::collect_stream_responses(&mut decode_stream, "Decode").await?;

    prefill.mark_completed();
    decode_stream.mark_completed();

    if merge_logprobs {
        merge_prefill_logprobs(&prefill_responses, &mut decode_responses);
    }

    Ok(decode_responses)
}

/// Merge prefill input_logprobs into decode responses
///
/// Takes input_logprobs from the first prefill response and copies them
/// into all decode responses. This is used in PD mode when logprobs are requested.
/// Only works with SGLang (vLLM doesn't support PD mode).
fn merge_prefill_logprobs(
    prefill_responses: &[ProtoGenerateComplete],
    decode_responses: &mut [ProtoGenerateComplete],
) {
    // Only SGLang supports PD mode and has input_logprobs
    if let Some(ProtoGenerateComplete::Sglang(prefill_first)) = prefill_responses.first() {
        if prefill_first.input_logprobs.is_some() {
            let _ = merge_input_logprobs(prefill_responses, decode_responses);
        }
    }
}

fn merge_input_logprobs(
    source_responses: &[ProtoGenerateComplete],
    decode_responses: &mut [ProtoGenerateComplete],
) -> bool {
    if let Some(ProtoGenerateComplete::Sglang(source_first)) = source_responses.first() {
        if let Some(input_logprobs) = source_first.input_logprobs.as_ref() {
            for response in decode_responses.iter_mut() {
                if let ProtoGenerateComplete::Sglang(decode_resp) = response {
                    decode_resp.input_logprobs = Some(input_logprobs.clone());
                }
            }
            return true;
        }
    }
    false
}
