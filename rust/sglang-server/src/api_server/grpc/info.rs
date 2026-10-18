//! Frontend metadata to the `api.v1` info responses.
//!
//! Both protos carry the JSON-typed members (`preferred_sampling_params`, the
//! allowlisted `internal_states`) as JSON text, so the gRPC reply holds the
//! same documents the HTTP reply embeds.

use sglang_api_types::api::v1 as api;

use crate::message::info::{ModelInfo, ServerInfo};

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
        launch_fields: _,
        model_path,
        served_model_name,
        tokenizer_path,
        max_context_length,
        max_total_num_tokens,
        version,
        frontend: _,
        startup_time: _,
        internal_states,
        host,
        port,
        incremental_streaming_output,
        disaggregation_mode,
        deployment_info,
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
        host,
        port: u32::from(port),
        incremental_streaming_output,
        disaggregation_mode: serde_json::to_value(disaggregation_mode)
            .map_err(|error| error.to_string())?
            .as_str()
            .unwrap_or("null")
            .to_owned(),
        deployment_info: Some(api::DeploymentInfo {
            disaggregation_bootstrap_port: deployment_info
                .disaggregation_bootstrap_port
                .map(u32::from),
            node_rank: deployment_info.node_rank,
            nnodes: deployment_info.nnodes,
            dp_size: deployment_info.dp_size,
            dp_rank: deployment_info.dp_rank,
            dist_init_addr: deployment_info.dist_init_addr,
            base_gpu_id: deployment_info.base_gpu_id,
            page_size: deployment_info.page_size,
            dcp_size: deployment_info.dcp_size,
            max_running_requests: deployment_info.max_running_requests,
            max_prefill_tokens: deployment_info.max_prefill_tokens,
            enable_dp_attention: deployment_info.enable_dp_attention,
            speculative_algorithm: deployment_info.speculative_algorithm,
            enable_hierarchical_cache: deployment_info.enable_hierarchical_cache,
            hicache_ratio: deployment_info.hicache_ratio,
            hicache_size: deployment_info.hicache_size,
            hicache_write_policy: deployment_info.hicache_write_policy,
            hicache_host_total_tokens: deployment_info.hicache_host_total_tokens,
            kv_event_sources: deployment_info
                .kv_event_sources
                .into_iter()
                .map(|source| api::KvEventSource {
                    dp_rank: source.dp_rank,
                    endpoint: source.endpoint,
                    topic: source.topic,
                    block_size: source.block_size,
                })
                .collect(),
        }),
    })
}
