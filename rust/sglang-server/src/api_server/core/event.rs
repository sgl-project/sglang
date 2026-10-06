//! The semantic events a call yields: incremental and terminal outputs, a
//! request-scoped failure, and the health probe's result. Transport-neutral
//! by design: no Axum, HTTP, SSE, Tonic, protobuf, or scheduler-channel
//! details.

use super::CoreError;
use crate::message::finish_reason::FinishReason;
use crate::message::response::{ChunkEvent, ChunkExtras};
use crate::message::types::TokenIds;

/// Canonical transport-neutral output payload for generation events.
///
/// The runtime correlation ID is deliberately absent: it is an implementation
/// detail used for scheduler routing, while adapters obtain the client-visible
/// identity from [`crate::api_server::core::CoreCall::public_id`]. Runtime-only
/// response variants such as raw control bytes are likewise hidden by
/// [`crate::api_server::core::CoreCall`].
#[derive(Clone, Debug, Default)]
pub(crate) struct CoreOutput {
    pub(crate) token_ids: TokenIds,
    pub(crate) finish_reason: Option<FinishReason>,
    pub(crate) prompt_tokens: u32,
    pub(crate) text: String,
    pub(crate) completion_tokens: u64,
    pub(crate) extras: Option<Box<ChunkExtras>>,
}

impl From<ChunkEvent> for CoreOutput {
    fn from(output: ChunkEvent) -> Self {
        let ChunkEvent {
            rid: _,
            token_ids,
            finish_reason,
            prompt_tokens,
            text,
            completion_tokens,
            extras,
        } = output;
        Self {
            token_ids,
            finish_reason,
            prompt_tokens,
            text,
            completion_tokens,
            extras,
        }
    }
}

impl CoreOutput {
    /// Fold one runtime delta into this cumulative native-generation output.
    ///
    /// Runtime events are always incremental. Native HTTP and `api.v1`
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

/// Semantic result of a deep core health check.
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
pub(crate) enum CoreEvent {
    /// One incremental model-output update.
    Delta(CoreOutput),
    /// The final model-output update for this request.
    Finished(CoreOutput),
    /// A request-scoped failure. Other requests in a batch may continue.
    Failed(CoreError),
}

impl CoreEvent {
    pub(crate) fn is_terminal(&self) -> bool {
        matches!(self, Self::Finished(_) | Self::Failed(_))
    }
}
