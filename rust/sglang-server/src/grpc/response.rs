//! Semantic frontend events to `runtime.v1` stream items and statuses.

use std::collections::HashMap;
use std::time::Duration;

use sglang_grpc_types::sglang::runtime::v1 as proto;
use tonic::{Code, Status};

use super::ResponseStream;
use crate::frontend::{
    FrontendCall, FrontendError, FrontendErrorKind, FrontendEvent, FrontendOutput,
};
use crate::native_generation::meta_info_value;

pub(super) fn text_generate_stream(
    call: FrontendCall,
    incremental: bool,
    response_timeout: Duration,
) -> ResponseStream<proto::TextGenerateResponse> {
    generation_stream(
        call,
        incremental,
        response_timeout,
        |output, meta_info, finished| {
            Ok(proto::TextGenerateResponse {
                text: output.text.clone(),
                meta_info,
                finished,
            })
        },
    )
}

pub(super) fn generate_stream(
    call: FrontendCall,
    incremental: bool,
    response_timeout: Duration,
) -> ResponseStream<proto::GenerateResponse> {
    generation_stream(
        call,
        incremental,
        response_timeout,
        |output, meta_info, finished| {
            Ok(proto::GenerateResponse {
                output_ids: output
                    .token_ids
                    .iter()
                    .copied()
                    .map(i32::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| {
                        Status::internal("output token ID exceeds runtime.v1 int32 range")
                    })?,
                meta_info,
                finished,
            })
        },
    )
}

fn generation_stream<T, F>(
    mut call: FrontendCall,
    incremental: bool,
    response_timeout: Duration,
    build: F,
) -> ResponseStream<T>
where
    T: Send + 'static,
    F: Fn(&FrontendOutput, HashMap<String, String>, bool) -> Result<T, Status> + Send + 'static,
{
    let public_id = call.public_id().to_owned();
    let stream = async_stream::stream! {
        let mut cumulative = FrontendOutput::default();
        let mut cumulative_completion_tokens = 0;
        loop {
            let event = match tokio::time::timeout(response_timeout, call.recv()).await {
                Ok(Some(event)) => event,
                // The loop stops on terminal events, so this is defensive only.
                Ok(None) => break,
                Err(_) => {
                    yield Err(Status::deadline_exceeded("Stream chunk timed out"));
                    break;
                }
            };

            let (mut delta, finished) = match event {
                FrontendEvent::Delta(output) => (output, false),
                FrontendEvent::Finished(output) => (output, true),
                FrontendEvent::Failed(error) => {
                    yield Err(status(error));
                    break;
                }
            };

            let output = if incremental {
                // Python's incremental native stream still reports the cumulative
                // completion-token count in every frame. Do not retain the full
                // generated output just to calculate that one cumulative field.
                cumulative_completion_tokens += delta.completion_tokens;
                delta.completion_tokens = cumulative_completion_tokens;
                &delta
            } else {
                cumulative.append_delta(&delta);
                &cumulative
            };
            let meta_info = grpc_meta_info(output, &public_id);
            match build(output, meta_info, finished) {
                Ok(response) => yield Ok(response),
                Err(error) => {
                    yield Err(error);
                    break;
                }
            }
            if finished {
                break;
            }
        }
        // `call` remains owned by this stream. Dropping a non-terminal stream or
        // timing out drops the armed FrontendCall and uses the core abort lane;
        // observing a terminal event has already disarmed it.
    };
    Box::pin(stream)
}

/// runtime.v1 represents heterogeneous metadata as strings. Match the existing
/// Python-backed server by JSON-encoding each value independently—including
/// string values such as the request ID.
fn grpc_meta_info(output: &FrontendOutput, public_id: &str) -> HashMap<String, String> {
    let serde_json::Value::Object(meta_info) = meta_info_value(output, public_id) else {
        unreachable!("native meta_info is always a JSON object");
    };
    meta_info
        .into_iter()
        .map(|(key, value)| (key, value.to_string()))
        .collect()
}

pub(super) fn status(error: FrontendError) -> Status {
    status_from_kind(error.kind(), error.to_string())
}

pub(super) fn status_from_kind(kind: FrontendErrorKind, message: impl Into<String>) -> Status {
    let code = match kind {
        FrontendErrorKind::InvalidArgument => Code::InvalidArgument,
        FrontendErrorKind::NotFound => Code::NotFound,
        FrontendErrorKind::FailedPrecondition => Code::FailedPrecondition,
        FrontendErrorKind::ResourceExhausted => Code::ResourceExhausted,
        FrontendErrorKind::Cancelled => Code::Cancelled,
        FrontendErrorKind::DeadlineExceeded => Code::DeadlineExceeded,
        FrontendErrorKind::Unavailable => Code::Unavailable,
        FrontendErrorKind::Internal => Code::Internal,
    };
    Status::new(code, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_errors_map_to_canonical_grpc_codes() {
        let cases = [
            (
                FrontendError::InvalidArgument("bad".into()),
                Code::InvalidArgument,
            ),
            (
                FrontendError::RuntimeRejected {
                    kind: FrontendErrorKind::NotFound,
                    message: "missing".into(),
                    legacy_http_status: 404,
                },
                Code::NotFound,
            ),
            (
                FrontendError::RuntimeRejected {
                    kind: FrontendErrorKind::FailedPrecondition,
                    message: "not ready".into(),
                    legacy_http_status: 412,
                },
                Code::FailedPrecondition,
            ),
            (
                FrontendError::Overloaded("full".into()),
                Code::ResourceExhausted,
            ),
            (FrontendError::Cancelled("gone".into()), Code::Cancelled),
            (
                FrontendError::RuntimeRejected {
                    kind: FrontendErrorKind::DeadlineExceeded,
                    message: "late".into(),
                    legacy_http_status: 504,
                },
                Code::DeadlineExceeded,
            ),
            (FrontendError::Unavailable, Code::Unavailable),
            (FrontendError::Internal("bug".into()), Code::Internal),
        ];

        for (error, expected) in cases {
            assert_eq!(status(error).code(), expected);
        }
    }
}
