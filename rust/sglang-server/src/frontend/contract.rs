//! Transport-neutral types shared by frontend adapters.
//!
//! These types describe what an inference operation means after a wire adapter
//! has normalized its request. They intentionally contain no Axum, HTTP, SSE,
//! Tonic, protobuf, or scheduler-channel details.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::message::config::{DisaggregationMode, PreferredSamplingParams};
use crate::message::finish_reason::FinishReason;
use crate::message::request::GenerateRequest;
use crate::message::response::{ChunkEvent, ChunkExtras};
use crate::message::types::TokenIds;

/// Canonical transport-neutral input for one generation operation.
///
/// The existing in-process request already contains semantic generation fields
/// and has no Serde, Axum, Tonic, or protobuf representation attached to it.
/// Re-exporting it through the frontend contract gives every adapter one public
/// boundary without duplicating the runtime's request model.
pub(crate) type FrontendRequest = GenerateRequest;

/// Canonical transport-neutral output payload for generation events.
///
/// The runtime correlation ID is deliberately absent: it is an implementation
/// detail used for scheduler routing, while adapters obtain the client-visible
/// identity from [`crate::frontend::FrontendCall::public_id`]. Runtime-only
/// response variants such as raw control bytes are likewise hidden by
/// [`crate::frontend::FrontendCall`].
#[derive(Clone, Debug, Default)]
pub(crate) struct FrontendOutput {
    pub(crate) token_ids: TokenIds,
    pub(crate) finish_reason: Option<FinishReason>,
    pub(crate) prompt_tokens: u32,
    pub(crate) text: String,
    pub(crate) completion_tokens: u64,
    pub(crate) stop_token_trimmed: bool,
    pub(crate) extras: Option<Box<ChunkExtras>>,
}

impl From<ChunkEvent> for FrontendOutput {
    fn from(output: ChunkEvent) -> Self {
        let ChunkEvent {
            rid: _,
            token_ids,
            finish_reason,
            prompt_tokens,
            text,
            completion_tokens,
            stop_token_trimmed,
            extras,
        } = output;
        Self {
            token_ids,
            finish_reason,
            prompt_tokens,
            text,
            completion_tokens,
            stop_token_trimmed,
            extras,
        }
    }
}

impl FrontendOutput {
    /// The token IDs behind `text`: `token_ids` without a matched stop token
    /// that the detokenizer left out of the text. Parsers that read IDs
    /// alongside the text use this view so the stop token stays out of their
    /// output too.
    pub(crate) fn stop_trimmed_token_ids(&self) -> &[i64] {
        let trimmed = usize::from(self.stop_token_trimmed);
        &self.token_ids[..self.token_ids.len().saturating_sub(trimmed)]
    }

    /// Fold one runtime delta into this cumulative native-generation output.
    ///
    /// Runtime events are always incremental. Native HTTP and `runtime.v1`
    /// gRPC independently decide whether to expose those deltas or the
    /// cumulative result, so the fold itself lives at their shared semantic
    /// boundary rather than in either wire adapter.
    pub(crate) fn append_delta(&mut self, delta: &Self) {
        self.text.push_str(&delta.text);
        self.token_ids.extend_from_slice(&delta.token_ids);
        self.completion_tokens += delta.completion_tokens;
        self.prompt_tokens = delta.prompt_tokens;
        if delta.finish_reason.is_some() {
            self.finish_reason = delta.finish_reason.clone();
            self.stop_token_trimmed = delta.stop_token_trimmed;
        }

        let Some(delta_extras) = delta.extras.as_deref() else {
            return;
        };
        let extras = self
            .extras
            .get_or_insert_with(|| Box::new(ChunkExtras::default()));

        extras
            .out_lp_val
            .extend_from_slice(&delta_extras.out_lp_val);
        extras
            .out_lp_idx
            .extend_from_slice(&delta_extras.out_lp_idx);
        extras
            .out_top_val
            .extend_from_slice(&delta_extras.out_top_val);
        extras
            .out_top_idx
            .extend_from_slice(&delta_extras.out_top_idx);
        extras
            .out_top_lens
            .extend_from_slice(&delta_extras.out_top_lens);
        extras
            .out_tid_val
            .extend_from_slice(&delta_extras.out_tid_val);
        extras
            .out_tid_idx
            .extend_from_slice(&delta_extras.out_tid_idx);
        extras
            .out_tid_lens
            .extend_from_slice(&delta_extras.out_tid_lens);
        extras
            .out_lp_txt
            .extend_from_slice(&delta_extras.out_lp_txt);
        extras
            .out_top_txt
            .extend_from_slice(&delta_extras.out_top_txt);
        extras
            .out_tid_txt
            .extend_from_slice(&delta_extras.out_tid_txt);

        // Input logprobs arrive once (during prefill). A later non-empty value
        // replaces the earlier one, matching the existing native HTTP fold.
        if !delta_extras.in_lp_val.is_empty() {
            extras.in_lp_val = delta_extras.in_lp_val.clone();
            extras.in_lp_idx = delta_extras.in_lp_idx.clone();
            extras.in_lp_txt = delta_extras.in_lp_txt.clone();
        }
        if !delta_extras.in_top_lens.is_empty() {
            extras.in_top_val = delta_extras.in_top_val.clone();
            extras.in_top_idx = delta_extras.in_top_idx.clone();
            extras.in_top_lens = delta_extras.in_top_lens.clone();
            extras.in_top_txt = delta_extras.in_top_txt.clone();
        }
        if !delta_extras.in_tid_lens.is_empty() {
            extras.in_tid_val = delta_extras.in_tid_val.clone();
            extras.in_tid_idx = delta_extras.in_tid_idx.clone();
            extras.in_tid_lens = delta_extras.in_tid_lens.clone();
            extras.in_tid_txt = delta_extras.in_tid_txt.clone();
        }
        // Hidden states are non-cumulative: the latest complete set wins.
        if !delta_extras.hidden_lens.is_empty() {
            extras.hidden_val = delta_extras.hidden_val.clone();
            extras.hidden_lens = delta_extras.hidden_lens.clone();
        }
    }
}

/// Semantic result of a deep frontend health check.
///
/// Expected lifecycle states are values rather than transport errors so each
/// adapter can render them according to its own protocol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HealthStatus {
    Healthy,
    NotReady,
    Stalled,
}

/// A semantic generation event before any transport-specific framing.
#[derive(Debug)]
pub(crate) enum FrontendEvent {
    /// One incremental model-output update.
    Delta(FrontendOutput),
    /// The final model-output update for this request.
    Finished(FrontendOutput),
    /// A request-scoped failure. Other requests in a batch may continue.
    Failed(FrontendError),
}

/// Protocol-independent error category used by transport adapters.
///
/// The set intentionally follows operation semantics rather than HTTP or gRPC
/// status spaces. An adapter maps this category into its own wire status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FrontendErrorKind {
    InvalidArgument,
    NotFound,
    FailedPrecondition,
    ResourceExhausted,
    Cancelled,
    DeadlineExceeded,
    Unavailable,
    Internal,
}

impl FrontendEvent {
    pub(crate) fn is_terminal(&self) -> bool {
        matches!(self, Self::Finished(_) | Self::Failed(_))
    }
}

/// A transport-neutral failure category.
///
/// HTTP and gRPC adapters map these categories into their own status spaces;
/// neither adapter needs to understand errors emitted by runtime stages.
#[derive(Clone, Debug, thiserror::Error)]
pub(crate) enum FrontendError {
    #[error("{0}")]
    InvalidArgument(String),

    /// The frontend intake loop has shut down. A caller may retry elsewhere.
    #[error("service unavailable")]
    Unavailable,

    /// Runtime capacity was exhausted before the request could make progress.
    #[error("{0}")]
    Overloaded(String),

    #[error("{0}")]
    Cancelled(String),

    /// The internal producer disappeared before returning a terminal result.
    ///
    /// This is a server-side failure, not evidence that the caller cancelled.
    #[error("response truncated before completion")]
    ResponseTruncated,

    /// A scheduler-side abort that carried an error status.
    ///
    /// The current Python scheduler communicates this category as an HTTP
    /// status code inside its finish reason. The semantic event is still
    /// `Failed`; retaining the legacy code only lets the HTTP adapter preserve
    /// existing behavior while another adapter maps it into its own status
    /// space. New frontend failures should use the semantic variants above.
    #[error("{message}")]
    RuntimeRejected {
        kind: FrontendErrorKind,
        message: String,
        legacy_http_status: u16,
    },

    /// The runtime replied with an unexpected or malformed result.
    #[error("{0}")]
    InvalidResponse(String),

    #[error("{0}")]
    Internal(String),
}

impl FrontendError {
    /// Semantic status for adapters that do not use the scheduler's legacy
    /// HTTP code space (notably the future gRPC adapter).
    pub(crate) fn kind(&self) -> FrontendErrorKind {
        match self {
            Self::InvalidArgument(_) => FrontendErrorKind::InvalidArgument,
            Self::Unavailable => FrontendErrorKind::Unavailable,
            Self::Overloaded(_) => FrontendErrorKind::ResourceExhausted,
            Self::Cancelled(_) => FrontendErrorKind::Cancelled,
            Self::ResponseTruncated => FrontendErrorKind::Internal,
            Self::RuntimeRejected { kind, .. } => *kind,
            Self::InvalidResponse(_) | Self::Internal(_) => FrontendErrorKind::Internal,
        }
    }

    /// Translate the scheduler's inherited HTTP-coded failure into a semantic
    /// category once, at the runtime boundary. HTTP can retain the exact legacy
    /// status while other adapters consume [`Self::kind`].
    pub(super) fn from_runtime_rejection(message: String, legacy_http_status: u16) -> Self {
        let kind = match legacy_http_status {
            404 => FrontendErrorKind::NotFound,
            408 | 504 => FrontendErrorKind::DeadlineExceeded,
            412 => FrontendErrorKind::FailedPrecondition,
            413 | 429 => FrontendErrorKind::ResourceExhausted,
            499 => FrontendErrorKind::Cancelled,
            502 | 503 => FrontendErrorKind::Unavailable,
            400..=499 => FrontendErrorKind::InvalidArgument,
            _ => FrontendErrorKind::Internal,
        };
        Self::RuntimeRejected {
            kind,
            message,
            legacy_http_status,
        }
    }
}

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
/// so extracting the typed frontend contract does not change public responses.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum MemoryMeasurement {
    Number(f64),
    String(String),
}

#[cfg(test)]
mod tests {
    use super::FrontendOutput;

    /// A trimmed stop token stays in the accumulated `token_ids` but out of the
    /// parser view, which must line up with the accumulated text. A stop token
    /// kept in the text (`no_stop_trim`) stays in both.
    #[test]
    fn stop_trimmed_token_ids_exclude_only_a_trimmed_stop_token() {
        for (stop_token_trimmed, expected) in [(true, &[1, 2][..]), (false, &[1, 2, 3][..])] {
            let mut output = FrontendOutput {
                token_ids: vec![1],
                ..Default::default()
            };
            assert_eq!(output.stop_trimmed_token_ids(), [1]);

            output.append_delta(&FrontendOutput {
                token_ids: vec![2, 3],
                finish_reason: serde_json::from_value(
                    serde_json::json!({"type": "stop", "matched": 3}),
                )
                .expect("finish reason must parse"),
                stop_token_trimmed,
                ..Default::default()
            });
            assert_eq!(output.token_ids, [1, 2, 3]);
            assert_eq!(output.stop_trimmed_token_ids(), expected);
        }
    }
}
