//! Frontend metadata to the `api.v1` info responses.
//!
//! Both protos carry the JSON-typed members (`preferred_sampling_params`, the
//! allowlisted `internal_states`) as JSON text, so the gRPC reply holds the
//! same documents the HTTP reply embeds.

use sglang_api_types::api::v1 as api;

use crate::frontend::{ModelInfo, ServerInfo};

pub(super) fn model_info(info: ModelInfo) -> Result<api::GetModelInfoResponse, String> {
    let ModelInfo {
        model_path,
        served_model_name,
        tokenizer_path,
        is_generation,
        preferred_sampling_params,
        weight_version,
        load_format,
        reasoning_parser,
        tool_call_parser,
        disaggregation_mode: _,
    } = info;
    let preferred_sampling_params = preferred_sampling_params
        .map(|params| serde_json::to_string(&params.0))
        .transpose()
        .map_err(|error| format!("preferred_sampling_params is not JSON: {error}"))?;
    Ok(api::GetModelInfoResponse {
        model_path,
        served_model_name,
        tokenizer_path,
        is_generation,
        preferred_sampling_params,
        weight_version,
        load_format,
        reasoning_parser,
        tool_call_parser,
    })
}

pub(super) fn server_info(info: ServerInfo) -> Result<api::GetServerInfoResponse, String> {
    let ServerInfo {
        model_path,
        served_model_name,
        tokenizer_path,
        max_context_length,
        max_total_num_tokens,
        version,
        frontend: _,
        internal_states,
    } = info;
    let max_context_length = u32::try_from(max_context_length)
        .map_err(|_| format!("max_context_length {max_context_length} exceeds uint32"))?;
    let internal_states = internal_states
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("internal_states are not JSON: {error}"))?;
    Ok(api::GetServerInfoResponse {
        model_path,
        served_model_name,
        tokenizer_path,
        max_context_length,
        max_total_num_tokens: Some(max_total_num_tokens),
        version,
        internal_states,
    })
}
