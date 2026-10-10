//! Public model and server metadata: the typed results of `/get_model_info`
//! and `/server_info` on every transport.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::config::{DisaggregationMode, PreferredSamplingParams};

/// Static model metadata shared by every public transport.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ModelInfo {
    pub(crate) model_path: String,
    pub(crate) served_model_name: String,
    pub(crate) tokenizer_path: String,
    pub(crate) is_generation: bool,
    pub(crate) preferred_sampling_params: Option<PreferredSamplingParams>,
    pub(crate) weight_version: Option<String>,
    pub(crate) load_format: Option<String>,
    pub(crate) reasoning_parser: Option<String>,
    pub(crate) tool_call_parser: Option<String>,
    pub(crate) disaggregation_mode: DisaggregationMode,
}

/// Public server metadata plus scheduler-owned runtime metrics.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ServerInfo {
    pub(crate) model_path: String,
    pub(crate) served_model_name: String,
    pub(crate) tokenizer_path: String,
    pub(crate) max_context_length: u64,
    pub(crate) max_total_num_tokens: u64,
    pub(crate) version: String,
    pub(crate) frontend: &'static str,
    pub(crate) internal_states: Vec<InternalState>,
}

/// Scheduler-owned runtime metrics used by the public server-info operation.
///
/// Deserializing into this allowlisted shape is intentional: the scheduler's
/// raw state also contains its full launch arguments, including credentials.
/// Unknown fields are discarded before any transport receives the result.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub(crate) struct InternalState {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_gen_throughput: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) memory_usage: Option<MemoryUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) effective_max_running_requests_per_dp: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) avg_spec_accept_length: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) step_time_dict: Option<BTreeMap<u64, Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) rust_mm_transport: Option<BTreeMap<String, u64>>,
}

/// Public subset of scheduler memory metrics.
///
/// This nested allowlist is intentional: adding a scheduler field does not
/// silently add different public data to HTTP before the gRPC contract can add
/// the same field. New metrics should be promoted here deliberately.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub(crate) struct MemoryUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) weight: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) kvcache: Option<MemoryMeasurement>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) startup_available: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) token_capacity: Option<u64>,
    pub(crate) token_capacity_swa: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) graph: Option<BTreeMap<String, f64>>,
}

/// The scheduler's KV-cache measurement can be a native float or a NumPy
/// scalar stringified by the MessagePack bridge. Preserve that representation
/// so extracting the typed contract does not change public responses.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum MemoryMeasurement {
    Number(f64),
    String(String),
}
