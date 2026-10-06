//! The semantic events a call yields (incremental and terminal outputs, a
//! request-scoped failure, the health probe's result) and the rendering of an
//! output into the `api.v1` response types. The generated schema is the wire
//! shape for both adapters: HTTP serializes it through serde, gRPC encodes it
//! as protobuf, so a column added here reaches both the same way. No Axum,
//! SSE, Tonic, or scheduler-channel details.

use sglang_api_types::api::v1 as api;

use super::CoreError;
use crate::message::finish_reason::{FinishKind, FinishReason, Matched};
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

impl CoreOutput {
    /// One `/generate` frame. `index` is the batch position (batch form only);
    /// `e2e_latency` rides the terminal frame only.
    pub(crate) fn frame(
        &self,
        id: &str,
        index: Option<u32>,
        e2e_latency: Option<f64>,
    ) -> api::GenerateResponse {
        api::GenerateResponse {
            text: self.text.clone(),
            meta_info: Some(self.meta_info(id, e2e_latency)),
            // Omitted while empty (a text-only frame).
            output_ids: (!self.token_ids.is_empty()).then(|| api::TokenIds {
                ids: self.token_ids.clone(),
            }),
            index,
        }
    }

    pub(crate) fn meta_info(&self, id: &str, e2e_latency: Option<f64>) -> api::GenerateMetaInfo {
        let mut meta = api::GenerateMetaInfo {
            id: id.to_owned(),
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            finish_reason: self.finish_reason.as_ref().map(finish_reason),
            e2e_latency,
            ..Default::default()
        };
        let Some(extras) = self.extras.as_deref() else {
            return meta;
        };

        // Python always sets input and output token logprobs together, including
        // the empty input list emitted by a disaggregated decode node.
        if !extras.out_lp_val.is_empty() || !extras.in_lp_val.is_empty() {
            meta.output_token_logprobs = Some(logprob_entries(
                &extras.out_lp_val,
                &extras.out_lp_idx,
                opt_texts(&extras.out_lp_txt),
            ));
            meta.input_token_logprobs = Some(logprob_entries(
                &extras.in_lp_val,
                &extras.in_lp_idx,
                opt_texts(&extras.in_lp_txt),
            ));
        }
        if !extras.out_top_lens.is_empty() {
            meta.output_top_logprobs = ragged_logprob_rows(
                &extras.out_top_val,
                &extras.out_top_idx,
                &extras.out_top_lens,
                opt_texts(&extras.out_top_txt),
            );
        }
        if !extras.in_top_lens.is_empty() {
            meta.input_top_logprobs = ragged_logprob_rows(
                &extras.in_top_val,
                &extras.in_top_idx,
                &extras.in_top_lens,
                opt_texts(&extras.in_top_txt),
            );
        }
        if !extras.out_tid_lens.is_empty() {
            meta.output_token_ids_logprobs = ragged_logprob_rows(
                &extras.out_tid_val,
                &extras.out_tid_idx,
                &extras.out_tid_lens,
                opt_texts(&extras.out_tid_txt),
            );
        }
        if !extras.in_tid_lens.is_empty() {
            meta.input_token_ids_logprobs = ragged_logprob_rows(
                &extras.in_tid_val,
                &extras.in_tid_idx,
                &extras.in_tid_lens,
                opt_texts(&extras.in_tid_txt),
            );
        }
        if !extras.hidden_lens.is_empty() {
            meta.hidden_states = hidden_state_rows(&extras.hidden_val, &extras.hidden_lens);
        }
        meta
    }
}

/// A decoded-text column becomes the entries' text source only when populated.
fn opt_texts(texts: &[String]) -> Option<&[String]> {
    (!texts.is_empty()).then_some(texts)
}

/// One `[logprob, token_id, text]` entry.
fn logprob_entry(value: f32, token_id: i32, text: Option<&String>) -> api::LogprobEntry {
    api::LogprobEntry {
        // The scheduler's NaN sentinel means "no logprob at this position".
        logprob: (!value.is_nan()).then(|| f64::from(value)),
        token_id: i64::from(token_id),
        text: text.cloned(),
    }
}

fn logprob_entries(
    values: &[f32],
    token_ids: &[i32],
    texts: Option<&[String]>,
) -> api::LogprobEntries {
    let entries = values
        .iter()
        .zip(token_ids)
        .enumerate()
        .map(|(j, (&value, &token_id))| {
            logprob_entry(value, token_id, texts.and_then(|texts| texts.get(j)))
        })
        .collect();
    api::LogprobEntries { entries }
}

/// Ragged top-k / requested-token rows: one per position, null where that
/// position's length is zero.
fn ragged_logprob_rows(
    values: &[f32],
    token_ids: &[i32],
    lengths: &[u32],
    texts: Option<&[String]>,
) -> Vec<api::NullableTopLogprobs> {
    let mut rows = Vec::with_capacity(lengths.len());
    let mut offset = 0usize;
    for &length in lengths {
        let length = length as usize;
        let row = (length > 0).then(|| {
            // Bounds checks turn malformed parallel columns into a shortened
            // row instead of panicking a transport worker.
            let entries = (offset..offset + length)
                .filter_map(|j| {
                    Some(logprob_entry(
                        *values.get(j)?,
                        *token_ids.get(j)?,
                        texts.and_then(|texts| texts.get(j)),
                    ))
                })
                .collect();
            api::TopLogprobRow { entries }
        });
        rows.push(api::NullableTopLogprobs { row });
        offset += length;
    }
    rows
}

/// Reshape flat hidden-state values and per-row lengths into rows.
fn hidden_state_rows(values: &[f32], lengths: &[u32]) -> Vec<api::HiddenStateRow> {
    let mut rows = Vec::with_capacity(lengths.len());
    let mut offset = 0usize;
    for &length in lengths {
        let length = length as usize;
        let values = values
            .get(offset..offset + length)
            .unwrap_or(&[])
            .iter()
            .copied()
            .map(f64::from)
            .collect();
        rows.push(api::HiddenStateRow { values });
        offset += length;
    }
    rows
}

fn finish_reason(reason: &FinishReason) -> api::FinishReason {
    use api::finish_reason::Kind;
    let kind = match reason {
        FinishReason::Known(FinishKind::Stop { matched }) => Kind::Stop(api::FinishStop {
            matched: matched.as_ref().map(matched_value),
        }),
        FinishReason::Known(FinishKind::Length { length }) => {
            Kind::Length(api::FinishLength { length: *length })
        }
        FinishReason::Known(FinishKind::Abort(abort)) => Kind::Abort(Box::new(api::FinishAbort {
            message: abort.message.clone(),
            status_code: abort.status_code.map(u32::from),
            err_type: abort.err_type.clone(),
        })),
        // The passthrough arm keeps the raw map as JSON text, the proto's own
        // encoding for a `type` this build does not know.
        FinishReason::Unknown(fields) => {
            Kind::Unknown(serde_json::to_string(fields.as_ref()).unwrap_or_default())
        }
    };
    api::FinishReason { kind: Some(kind) }
}

fn matched_value(matched: &Matched) -> api::Matched {
    use api::matched::Value;
    let value = match matched {
        Matched::Token(token) => Value::Token(*token),
        Matched::Str(text) => Value::Str(text.clone()),
        Matched::Tokens(ids) => Value::Tokens(api::MatchedTokens { ids: ids.clone() }),
    };
    api::Matched { value: Some(value) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::response::ChunkExtras;

    fn json<T: serde::Serialize>(value: &T) -> serde_json::Value {
        serde_json::to_value(value).unwrap()
    }

    /// Python parity pinned on the typed shape: a NaN logprob is a null slot,
    /// input logprobs are set together with output ones, and a zero-length
    /// top-k position is a null row rather than an empty one.
    #[test]
    fn meta_info_keeps_the_native_null_conventions() {
        let extras = ChunkExtras {
            out_lp_val: vec![f32::NAN, -0.5],
            out_lp_idx: vec![7, 8],
            out_top_val: vec![-0.1],
            out_top_idx: vec![9],
            out_top_lens: vec![0, 1],
            ..Default::default()
        };
        let output = CoreOutput {
            extras: Some(Box::new(extras)),
            ..Default::default()
        };

        let meta = output.meta_info("rid", None);
        let entries = meta.output_token_logprobs.unwrap().entries;
        assert_eq!(entries[0].logprob, None);
        assert_eq!(entries[1].logprob, Some(f64::from(-0.5_f32)));
        assert_eq!(entries[1].token_id, 8);
        assert_eq!(meta.input_token_logprobs.unwrap().entries, []);
        assert!(meta.output_top_logprobs[0].row.is_none());
        let row = meta.output_top_logprobs[1].row.as_ref().unwrap();
        assert_eq!(row.entries[0].token_id, 9);
        assert!(meta.input_top_logprobs.is_empty());
    }

    /// The JSON the schema renders is SGLang's `[logprob, token_id, text]`
    /// tuple list, with the text slot null unless `return_text_in_logprobs`
    /// supplied a text column, and null positions in the ragged families.
    #[test]
    fn logprob_json_keeps_the_tuple_shape() {
        let texts = vec!["a".to_string(), "b".to_string()];
        assert_eq!(
            json(&logprob_entries(&[-0.5, -1.5], &[10, 20], None)),
            serde_json::json!([[-0.5f32, 10, null], [-1.5f32, 20, null]])
        );
        assert_eq!(
            json(&logprob_entries(&[-0.5, -1.5], &[10, 20], Some(&texts))),
            serde_json::json!([[-0.5f32, 10, "a"], [-1.5f32, 20, "b"]])
        );
        assert_eq!(
            json(&ragged_logprob_rows(&[-0.3], &[9], &[0, 1], None)),
            serde_json::json!([null, [[-0.3f32, 9, null]]])
        );
        // The NaN sentinel (the Python `None` logprob for the first prompt
        // token) is a null logprob with its token id preserved.
        assert_eq!(
            json(&ragged_logprob_rows(&[f32::NAN], &[7], &[1], None)),
            serde_json::json!([[[null, 7, null]]])
        );
        assert!(opt_texts(&[]).is_none());
        assert_eq!(opt_texts(&texts), Some(texts.as_slice()));
    }

    /// A finish type this build does not know still reaches the client, as
    /// the JSON text the proto's passthrough arm carries.
    #[test]
    fn unknown_finish_reason_passes_through_as_json_text() {
        let reason: FinishReason =
            serde_json::from_value(serde_json::json!({"type": "novel", "detail": 3})).unwrap();
        let api::FinishReason {
            kind: Some(api::finish_reason::Kind::Unknown(text)),
        } = finish_reason(&reason)
        else {
            panic!("unknown type takes the passthrough arm");
        };
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text).unwrap(),
            serde_json::json!({"type": "novel", "detail": 3})
        );
    }
}
