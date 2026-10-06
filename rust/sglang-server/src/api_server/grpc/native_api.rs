//! Core events to `api.v1` stream items and statuses.

use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use sglang_api_types::api::v1 as api;
use tonic::{Code, Status};

use super::service::ResponseStream;
use crate::api_server::core::{
    CoreCall, CoreError, CoreErrorKind, CoreEvent, CoreOutput, recv_indexed,
};

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
        item: Some(api::generate_stream_item::Item::Frame(output.frame(
            public_id,
            index,
            e2e_latency,
        ))),
    }
}

/// One item's failure inside the stream. `code` is the native error body's
/// HTTP status, as on the SSE error frame.
fn error_item(error: &CoreError, index: Option<u32>) -> api::GenerateStreamItem {
    api::GenerateStreamItem {
        item: Some(api::generate_stream_item::Item::Error(
            error.stream_error(index),
        )),
    }
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
}
