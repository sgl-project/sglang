//! Frontend events to `api.v1` stream items and statuses.
//!
//! The typed [`api::GenerateMetaInfo`] built here and the native HTTP
//! `meta_info` JSON (`http::frame::meta_info_value`) read the same
//! output columns with the same gating; a column added to one is added to
//! the other.

use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use sglang_api_types::api::v1 as api;
use tonic::{Code, Status};

use super::service::ResponseStream;
use crate::api_server::core::{
    CoreCall, CoreError, CoreErrorKind, CoreEvent, CoreOutput, recv_indexed,
};
use crate::message::finish_reason::{FinishKind, FinishReason, Matched};

pub(super) struct StreamOptions {
    /// `stream` as sent: false folds each item to its one terminal frame.
    pub stream: bool,
    /// Streamed frames are deltas (true) or cumulative (false, the default).
    pub incremental: bool,
    /// Batch form: every item carries its position.
    pub with_index: bool,
    pub response_timeout: Duration,
    pub created_at: Instant,
}

/// Multiplex the admitted calls into one `GenerateStreamItem` stream, the
/// gRPC twin of the native SSE multiplexer. Each call aborts itself if the
/// stream is dropped while it is unfinished.
pub(super) fn generate_stream(
    calls: Vec<CoreCall>,
    options: StreamOptions,
) -> ResponseStream<api::GenerateStreamItem> {
    let stream = async_stream::stream! {
        let public_ids: Vec<String> = calls
            .iter()
            .map(|call| call.public_id().to_owned())
            .collect();
        let mut cumulative: Vec<CoreOutput> =
            calls.iter().map(|_| CoreOutput::default()).collect();
        let index = |i: usize| options.with_index.then_some(i as u32);

        // Poll all receivers concurrently; re-arm a receiver's future after each
        // non-terminal event so its stream keeps flowing.
        let mut pending = FuturesUnordered::new();
        for (i, call) in calls.into_iter().enumerate() {
            pending.push(recv_indexed(i, call));
        }

        loop {
            let next = tokio::time::timeout(options.response_timeout, pending.next()).await;
            let (i, call, events) = match next {
                Ok(Some(next)) => next,
                // Every call has reached its terminal event.
                Ok(None) => break,
                Err(_) => {
                    yield Err(Status::deadline_exceeded("Stream chunk timed out"));
                    break;
                }
            };
            if events.is_empty() {
                // Defensive fallback: CoreCall normally turns a premature
                // runtime close into a Failed event.
                yield Ok(error_item(&CoreError::ResponseTruncated, index(i)));
                continue;
            }

            let acc = &mut cumulative[i];
            // Cumulative frames supersede one another, so a drained backlog
            // collapses to its last (Python's `out_list[-1]`); deltas can't be
            // dropped.
            let mut coalesced = false;
            let mut terminal = None;
            let mut failed = None;
            for event in events {
                match event {
                    CoreEvent::Delta(mut delta) => {
                        acc.append_delta(&delta);
                        if !options.stream {
                            continue;
                        }
                        if options.incremental {
                            // Python's incremental stream still reports the
                            // cumulative completion-token count in every frame.
                            delta.completion_tokens = acc.completion_tokens;
                            yield Ok(frame_item(&delta, &public_ids[i], index(i), None));
                        } else {
                            coalesced = true;
                        }
                    }
                    CoreEvent::Finished(delta) => {
                        acc.append_delta(&delta);
                        terminal = Some(delta);
                    }
                    CoreEvent::Failed(error) => failed = Some(error),
                }
            }

            if let Some(error) = failed {
                yield Ok(error_item(&error, index(i)));
            } else if let Some(mut delta) = terminal {
                let e2e_latency = Some(options.created_at.elapsed().as_secs_f64());
                // A unary reply is the cumulative result whatever the stream policy.
                let output = if options.stream && options.incremental {
                    delta.completion_tokens = acc.completion_tokens;
                    &delta
                } else {
                    &*acc
                };
                yield Ok(frame_item(output, &public_ids[i], index(i), e2e_latency));
            } else {
                if coalesced {
                    yield Ok(frame_item(acc, &public_ids[i], index(i), None));
                }
                pending.push(recv_indexed(i, call));
            }
        }
        // Unfinished calls are still owned by `pending`: dropping the stream
        // drops them and uses the core abort lane; a terminal event has
        // already disarmed its call.
    };
    Box::pin(stream)
}

fn frame_item(
    output: &CoreOutput,
    public_id: &str,
    index: Option<u32>,
    e2e_latency: Option<f64>,
) -> api::GenerateStreamItem {
    api::GenerateStreamItem {
        item: Some(api::generate_stream_item::Item::Frame(
            api::GenerateResponse {
                text: output.text.clone(),
                meta_info: Some(meta_info(output, public_id, e2e_latency)),
                output_ids: (!output.token_ids.is_empty()).then(|| api::TokenIds {
                    ids: output.token_ids.clone(),
                }),
                index,
            },
        )),
    }
}

/// One item's failure inside the stream. `code` is the native error body's
/// HTTP status, as on the SSE error frame.
fn error_item(error: &CoreError, index: Option<u32>) -> api::GenerateStreamItem {
    api::GenerateStreamItem {
        item: Some(api::generate_stream_item::Item::Error(
            api::GenerateStreamError {
                error: Some(api::ErrorBody {
                    message: error.to_string(),
                    code: u32::from(error.http_status().as_u16()),
                }),
                index,
            },
        )),
    }
}

fn meta_info(out: &CoreOutput, id: &str, e2e_latency: Option<f64>) -> api::GenerateMetaInfo {
    let mut meta = api::GenerateMetaInfo {
        id: id.to_owned(),
        prompt_tokens: out.prompt_tokens,
        completion_tokens: out.completion_tokens,
        finish_reason: out.finish_reason.as_ref().map(finish_reason),
        e2e_latency,
        ..Default::default()
    };
    let Some(extras) = out.extras.as_deref() else {
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

/// A decoded-text column becomes the entries' text source only when populated.
fn opt_texts(texts: &[String]) -> Option<&[String]> {
    (!texts.is_empty()).then_some(texts)
}

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

pub(super) fn status(error: CoreError) -> Status {
    let code = match error.kind() {
        CoreErrorKind::InvalidArgument => Code::InvalidArgument,
        CoreErrorKind::NotFound => Code::NotFound,
        CoreErrorKind::FailedPrecondition => Code::FailedPrecondition,
        CoreErrorKind::ResourceExhausted => Code::ResourceExhausted,
        CoreErrorKind::Cancelled => Code::Cancelled,
        CoreErrorKind::DeadlineExceeded => Code::DeadlineExceeded,
        CoreErrorKind::Unavailable => Code::Unavailable,
        CoreErrorKind::Internal => Code::Internal,
    };
    Status::new(code, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::response::ChunkExtras;

    #[test]
    fn semantic_errors_map_to_canonical_grpc_codes() {
        let cases = [
            (
                CoreError::InvalidArgument("bad".into()),
                Code::InvalidArgument,
            ),
            (
                CoreError::RuntimeRejected {
                    kind: CoreErrorKind::NotFound,
                    message: "missing".into(),
                    legacy_http_status: 404,
                },
                Code::NotFound,
            ),
            (
                CoreError::RuntimeRejected {
                    kind: CoreErrorKind::FailedPrecondition,
                    message: "not ready".into(),
                    legacy_http_status: 412,
                },
                Code::FailedPrecondition,
            ),
            (
                CoreError::Overloaded("full".into()),
                Code::ResourceExhausted,
            ),
            (CoreError::Cancelled("gone".into()), Code::Cancelled),
            (
                CoreError::RuntimeRejected {
                    kind: CoreErrorKind::DeadlineExceeded,
                    message: "late".into(),
                    legacy_http_status: 504,
                },
                Code::DeadlineExceeded,
            ),
            (CoreError::Unavailable, Code::Unavailable),
            (CoreError::Internal("bug".into()), Code::Internal),
        ];

        for (error, expected) in cases {
            assert_eq!(status(error).code(), expected);
        }
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

        let meta = meta_info(&output, "rid", None);
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
