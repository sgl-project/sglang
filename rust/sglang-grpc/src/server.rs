use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;

use pyo3::PyErr;
use pyo3::Python;
use pyo3::exceptions::{PyTypeError, PyValueError};
use tokio::sync::{Notify, mpsc::Receiver};
use tokio::time::{Duration, timeout};
use tokio_stream::Stream;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use crate::bridge::{
    PyBridge, RequestKey, ResponseChunk, ResponseMetadata, ResponseMetadataMode, SubmittedRequest,
    TerminalError,
};
use crate::proto;
use crate::utils::{
    build_classify_dict, build_embed_dict, build_generate_dict, build_text_embed_dict,
    build_text_generate_dict, extract_model_path,
};

pub struct SglangServiceImpl {
    pub bridge: Arc<PyBridge>,
    pub response_timeout: Duration,
}

type StreamResult<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;
pub const DEFAULT_RESPONSE_TIMEOUT_SECS: u64 = 300;

/// 64 MiB — leaves headroom for multimodal inputs and OpenAI JSON pass-through bodies,
/// well above tonic's 4 MiB decode default.
pub const DEFAULT_GRPC_MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

/// Resolve the per-message size cap (bytes) applied to the Tonic encoder/decoder.
//
// TODO(grpc-args): promote SGLANG_TONIC_PAYLOAD to a proper `--grpc-max-message-size`
// server argument once the launcher PR (3/4) wires server args through.
fn resolve_max_message_size() -> usize {
    match std::env::var("SGLANG_TONIC_PAYLOAD") {
        Ok(raw) => match raw.parse::<usize>() {
            Ok(n) if n > 0 => {
                tracing::info!(
                    bytes = n,
                    "Using SGLANG_TONIC_PAYLOAD override for gRPC max message size"
                );
                n
            }
            _ => {
                tracing::warn!(
                    value = %raw,
                    default = DEFAULT_GRPC_MAX_MESSAGE_SIZE,
                    "Ignoring invalid SGLANG_TONIC_PAYLOAD; using default"
                );
                DEFAULT_GRPC_MAX_MESSAGE_SIZE
            }
        },
        Err(_) => DEFAULT_GRPC_MAX_MESSAGE_SIZE,
    }
}

/// Classify a bridge `PyErr` into the right gRPC `Status`.
///
/// `PyValueError` / `PyTypeError` mean the client sent bad input — surface as
/// `INVALID_ARGUMENT` so callers can distinguish them from server failures.
/// Everything else (typically `PyRuntimeError`, but also Python tracebacks
/// from inside the tokenizer manager) maps to `INTERNAL`.
fn pyerr_to_status(err: PyErr, context: &str) -> Status {
    let is_client_error = Python::attach(|py| {
        err.is_instance_of::<PyValueError>(py) || err.is_instance_of::<PyTypeError>(py)
    });
    let msg = format!("{}: {}", context, err);
    if is_client_error {
        Status::invalid_argument(msg)
    } else {
        Status::internal(msg)
    }
}

async fn recv_chunk_with_timeout(
    receiver: &mut Receiver<ResponseChunk>,
    response_timeout: Duration,
    timeout_message: impl FnOnce() -> String,
) -> Result<Option<ResponseChunk>, Status> {
    timeout(response_timeout, receiver.recv())
        .await
        .map_err(|_| Status::deadline_exceeded(timeout_message()))
}

struct RequestAbortGuard {
    bridge: Arc<PyBridge>,
    key: RequestKey,
    armed: bool,
}

impl RequestAbortGuard {
    fn new(bridge: Arc<PyBridge>, key: RequestKey) -> Self {
        Self {
            bridge,
            key,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }

    fn abort_now(&mut self) {
        if self.armed {
            self.armed = false;
            spawn_abort(self.bridge.clone(), self.key.clone());
        }
    }
}

impl Drop for RequestAbortGuard {
    fn drop(&mut self) {
        if self.armed {
            // Dropping a response stream means the client stopped consuming; propagate
            // cancellation to Python without blocking the Tokio worker.
            spawn_abort(self.bridge.clone(), self.key.clone());
        }
    }
}

fn spawn_abort(bridge: Arc<PyBridge>, key: RequestKey) {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            // Fire-and-forget: dropping the JoinHandle detaches the task.
            drop(handle.spawn_blocking(move || {
                let _ = bridge.abort_request(&key);
            }));
        }
        Err(_) => {
            tracing::warn!(
                rid = key.rid(),
                "Skipping gRPC request abort because no Tokio runtime is available"
            );
        }
    }
}

async fn recv_terminal_chunk_for_request(
    bridge: &Arc<PyBridge>,
    key: &RequestKey,
    receiver: &mut Receiver<ResponseChunk>,
    response_timeout: Duration,
) -> Result<ResponseChunk, Status> {
    let mut abort_guard = RequestAbortGuard::new(bridge.clone(), key.clone());

    match recv_chunk_with_timeout(receiver, response_timeout, || {
        format!("Request timed out after {}s", response_timeout.as_secs())
    })
    .await
    {
        Ok(Some(ResponseChunk::Data(_))) => {
            tracing::warn!(
                rid = key.rid(),
                "Unary gRPC response received non-terminal Data chunk; expected Finished"
            );
            abort_guard.abort_now();
            Err(Status::internal(
                "Unary response protocol violation: expected Finished, got Data",
            ))
        }
        Ok(Some(chunk @ (ResponseChunk::Finished(_) | ResponseChunk::Error(_)))) => {
            abort_guard.disarm();
            Ok(chunk)
        }
        Ok(None) => {
            let (status, should_abort) = closed_stream_status(bridge, key);
            if should_abort {
                abort_guard.abort_now();
            } else {
                abort_guard.disarm();
            }
            Err(status)
        }
        Err(status) => {
            if status.code() == tonic::Code::DeadlineExceeded {
                abort_guard.abort_now();
            } else {
                abort_guard.disarm();
            }
            Err(status)
        }
    }
}

fn closed_stream_status(bridge: &Arc<PyBridge>, key: &RequestKey) -> (Status, bool) {
    if let Some(error) = bridge.take_terminal_error(key) {
        (terminal_error_status(error), false)
    } else {
        (
            Status::internal("gRPC response stream closed before a terminal response"),
            true,
        )
    }
}

fn terminal_error_status(error: TerminalError) -> Status {
    let message = error.message();
    match error {
        TerminalError::ChannelFull { .. } => Status::resource_exhausted(message),
        TerminalError::ClientDisconnected { .. } | TerminalError::Aborted { .. } => {
            Status::cancelled(message)
        }
    }
}

fn openai_status_code(meta_info: &HashMap<String, String>, default: i32) -> i32 {
    meta_info
        .get("status_code")
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default)
}

fn legacy_metadata(meta_info: ResponseMetadata) -> Result<HashMap<String, String>, Box<Status>> {
    meta_info
        .into_legacy()
        .map_err(|error| Box::new(Status::internal(error)))
}

fn json_to_prost_value(value: serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(value) => Kind::BoolValue(value),
        serde_json::Value::Number(value) => Kind::NumberValue(value.as_f64().unwrap_or_default()),
        serde_json::Value::String(value) => Kind::StringValue(value),
        serde_json::Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values.into_iter().map(json_to_prost_value).collect(),
        }),
        serde_json::Value::Object(fields) => Kind::StructValue(prost_types::Struct {
            fields: fields
                .into_iter()
                .map(|(key, value)| (key, json_to_prost_value(value)))
                .collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

const TYPED_META_KEYS: &[&str] = &[
    "id",
    "finish_reason",
    "prompt_tokens",
    "input_tokens",
    "completion_tokens",
    "output_tokens",
    "cached_tokens",
    "cached_prompt_tokens",
    "input_token_logprobs",
    "output_token_logprobs",
    "input_top_logprobs",
    "output_top_logprobs",
    "output_token_logprobs_length",
];

const MAX_TYPED_GENERATION_CHOICES: i32 = 1024;

fn expected_generation_choices(request: &proto::GenerateRequest) -> Result<usize, Box<Status>> {
    let choices = request
        .sampling_params
        .as_ref()
        .and_then(|params| params.n)
        .unwrap_or(1);
    if !(1..=MAX_TYPED_GENERATION_CHOICES).contains(&choices) {
        return Err(Box::new(Status::invalid_argument(format!(
            "sampling_params.n must be between 1 and {MAX_TYPED_GENERATION_CHOICES}, got {choices}"
        ))));
    }
    Ok(choices as usize)
}

fn engine_metadata(
    meta: &serde_json::Map<String, serde_json::Value>,
) -> Option<prost_types::Struct> {
    let fields = meta
        .iter()
        .filter(|(key, _)| !TYPED_META_KEYS.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), json_to_prost_value(value.clone())))
        .collect::<std::collections::BTreeMap<_, _>>();
    (!fields.is_empty()).then_some(prost_types::Struct { fields })
}

fn meta_u64(meta: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> u64 {
    keys.iter()
        .find_map(|key| meta.get(*key).and_then(serde_json::Value::as_u64))
        .unwrap_or_default()
}

fn has_finish_reason(meta: &serde_json::Map<String, serde_json::Value>) -> bool {
    meta.get("finish_reason")
        .is_some_and(|value| !value.is_null())
}

fn logprob_entry(
    selected: &serde_json::Value,
    top: Option<&serde_json::Value>,
) -> Result<Option<proto::TokenLogprob>, String> {
    let Some(parts) = selected.as_array() else {
        return Err("SGLang returned a non-array logprob entry".into());
    };
    if parts.first().is_none_or(serde_json::Value::is_null) {
        return Ok(None);
    }
    let logprob = parts
        .first()
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| "SGLang logprob entry is missing a numeric logprob".to_string())?
        as f32;
    let token_id = parts
        .get(1)
        .and_then(serde_json::Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| "SGLang logprob entry is missing a valid token id".to_string())?;
    let text = parts
        .get(2)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let top_logprobs = match top {
        Some(serde_json::Value::Array(entries)) => entries
            .iter()
            .enumerate()
            .map(|(rank, entry)| {
                let parts = entry
                    .as_array()
                    .ok_or_else(|| "SGLang returned a non-array top-logprob entry".to_string())?;
                Ok(proto::LogprobAlternative {
                    logprob: parts
                        .first()
                        .and_then(serde_json::Value::as_f64)
                        .ok_or_else(|| "top-logprob entry is missing a logprob".to_string())?
                        as f32,
                    token_id: parts
                        .get(1)
                        .and_then(serde_json::Value::as_i64)
                        .and_then(|value| i32::try_from(value).ok())
                        .ok_or_else(|| "top-logprob entry is missing a token id".to_string())?,
                    text: parts
                        .get(2)
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned),
                    rank: i32::try_from(rank + 1).ok(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        Some(_) => return Err("SGLang returned non-array top logprobs".into()),
        None => Vec::new(),
    };
    Ok(Some(proto::TokenLogprob {
        logprob,
        token_id,
        text,
        top_logprobs,
    }))
}

struct ChoiceTracker {
    expected: usize,
    terminal: HashSet<i32>,
}

impl ChoiceTracker {
    fn new(expected: usize) -> Self {
        Self {
            expected,
            terminal: HashSet::new(),
        }
    }

    fn observe(
        &mut self,
        choice_index: i32,
        choice_terminal: bool,
        request_finished: bool,
        rpc_name: &str,
    ) -> Result<bool, String> {
        if choice_index < 0 || choice_index as usize >= self.expected {
            return Err(format!(
                "SGLang returned choice index {choice_index} outside 0..{}",
                self.expected
            ));
        }
        if self.terminal.contains(&choice_index) {
            return Err(format!(
                "SGLang returned data after terminal for choice {choice_index}"
            ));
        }
        if choice_terminal {
            self.terminal.insert(choice_index);
        }
        if request_finished && self.terminal.len() != self.expected {
            return Err(format!(
                "SGLang closed {rpc_name} after {}/{} terminal choices",
                self.terminal.len(),
                self.expected
            ));
        }
        Ok(self.terminal.len() == self.expected)
    }
}

fn logprobs_from_meta(
    meta: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<proto::Logprobs>, String> {
    let output_values: &[serde_json::Value] = match meta.get("output_token_logprobs") {
        Some(serde_json::Value::Array(values)) => values.as_slice(),
        Some(_) => return Err("SGLang returned non-array output logprobs".into()),
        None => &[],
    };
    let top_values = match meta.get("output_top_logprobs") {
        Some(serde_json::Value::Array(values)) => Some(values),
        Some(_) => return Err("SGLang returned non-array output top logprobs".into()),
        None => None,
    };
    let output = output_values
        .iter()
        .enumerate()
        .map(|(index, selected)| {
            logprob_entry(selected, top_values.and_then(|values| values.get(index)))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let prompt_values: &[serde_json::Value] = match meta.get("input_token_logprobs") {
        Some(serde_json::Value::Array(values)) => values.as_slice(),
        Some(_) => return Err("SGLang returned non-array prompt logprobs".into()),
        None => &[],
    };
    let prompt_top = match meta.get("input_top_logprobs") {
        Some(serde_json::Value::Array(values)) => Some(values),
        Some(_) => return Err("SGLang returned non-array prompt top logprobs".into()),
        None => None,
    };
    let prompt = prompt_values
        .iter()
        .enumerate()
        .map(|(index, selected)| {
            logprob_entry(selected, prompt_top.and_then(|values| values.get(index)))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    Ok((!output.is_empty() || !prompt.is_empty()).then_some(proto::Logprobs { output, prompt }))
}

enum TypedTerminal {
    Finish(proto::GenerationFinish),
    Error(proto::GenerationError),
}

fn generation_terminal(meta: &serde_json::Map<String, serde_json::Value>) -> TypedTerminal {
    let Some(value) = meta.get("finish_reason").filter(|value| !value.is_null()) else {
        return TypedTerminal::Error(proto::GenerationError {
            code: proto::GenerationErrorCode::Internal as i32,
            message: "SGLang stream ended without finish_reason".into(),
            retryable: false,
        });
    };
    let finish_type = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .or_else(|| value.as_str())
        .unwrap_or("error");
    if matches!(finish_type, "abort" | "error") {
        let status_code = value.get("status_code").and_then(serde_json::Value::as_i64);
        let (code, retryable) = match status_code {
            Some(408 | 504) => (proto::GenerationErrorCode::DeadlineExceeded, true),
            Some(499) => (proto::GenerationErrorCode::Cancelled, false),
            Some(400..=499) => (proto::GenerationErrorCode::InvalidArgument, false),
            Some(503) => (proto::GenerationErrorCode::Unavailable, true),
            None if finish_type == "abort" => (proto::GenerationErrorCode::Cancelled, false),
            _ => (proto::GenerationErrorCode::Internal, false),
        };
        return TypedTerminal::Error(proto::GenerationError {
            code: code as i32,
            message: value
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("SGLang generation failed")
                .to_string(),
            retryable,
        });
    }
    let reason = match finish_type {
        "stop" => proto::FinishReason::Stop,
        "length" => proto::FinishReason::Length,
        "cancelled" => proto::FinishReason::Cancelled,
        _ => proto::FinishReason::Unspecified,
    };
    let stop_reason = value.get("matched").and_then(|matched| {
        use proto::stop_reason::Reason;
        let reason = if let Some(value) = matched.as_str() {
            Some(Reason::MatchedString(value.to_string()))
        } else {
            matched
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .map(Reason::MatchedTokenId)
        }?;
        Some(proto::StopReason {
            reason: Some(reason),
        })
    });
    TypedTerminal::Finish(proto::GenerationFinish {
        reason: reason as i32,
        stop_reason,
    })
}

struct TypedGenerationChunk {
    choice_index: i32,
    delta_output_ids: Vec<i32>,
    logprobs: Option<proto::Logprobs>,
    usage: Option<proto::Usage>,
    engine_metadata: Option<prost_types::Struct>,
    terminal: Option<TypedTerminal>,
}

impl TypedGenerationChunk {
    fn into_response(self, request_id: String) -> proto::TypedGenerateResponse {
        let terminal = self.terminal.map(|terminal| match terminal {
            TypedTerminal::Finish(finish) => {
                proto::typed_generate_response::Terminal::Finish(finish)
            }
            TypedTerminal::Error(error) => proto::typed_generate_response::Terminal::Error(error),
        });
        proto::TypedGenerateResponse {
            request_id,
            delta_output_ids: self.delta_output_ids,
            choice_index: self.choice_index,
            logprobs: self.logprobs,
            usage: self.usage,
            engine_metadata: self.engine_metadata,
            terminal,
        }
    }
}

fn typed_generation_chunk(
    data: crate::bridge::ResponseData,
    choice_terminal: bool,
) -> Result<TypedGenerationChunk, String> {
    let crate::bridge::ResponseData {
        output_ids,
        choice_index,
        meta_info,
        ..
    } = data;
    let meta_info = meta_info.as_typed()?;
    let delta_output_ids = output_ids.unwrap_or_default();
    let logprobs = logprobs_from_meta(meta_info)?;
    let usage_keys = [
        "prompt_tokens",
        "input_tokens",
        "completion_tokens",
        "output_tokens",
        "cached_tokens",
        "cached_prompt_tokens",
    ];
    let usage = (choice_terminal && usage_keys.iter().any(|key| meta_info.contains_key(*key)))
        .then(|| {
            let prompt_tokens = meta_u64(meta_info, &["prompt_tokens", "input_tokens"]);
            let completion_tokens = meta_u64(meta_info, &["completion_tokens", "output_tokens"]);
            proto::Usage {
                prompt_tokens,
                completion_tokens,
                cached_prompt_tokens: meta_u64(
                    meta_info,
                    &["cached_tokens", "cached_prompt_tokens"],
                ),
            }
        });
    Ok(TypedGenerationChunk {
        choice_index,
        delta_output_ids,
        logprobs,
        usage,
        engine_metadata: engine_metadata(meta_info),
        terminal: choice_terminal.then(|| generation_terminal(meta_info)),
    })
}

#[tonic::async_trait]
impl proto::sglang_service_server::SglangService for SglangServiceImpl {
    // --- SGLang-native RPCs: TextGenerate / Generate ---

    type TextGenerateStream = StreamResult<proto::TextGenerateResponse>;

    async fn text_generate(
        &self,
        request: Request<proto::TextGenerateRequest>,
    ) -> Result<Response<Self::TextGenerateStream>, Status> {
        let req = request.into_inner();
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_text_generate_dict(&rid, &req).map_err(Status::invalid_argument)?;

        let SubmittedRequest { key, mut receiver } = self
            .bridge
            .submit_request(
                &rid,
                "generate",
                req_dict,
                ResponseMetadataMode::LegacyStringMap,
            )
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let bridge = self.bridge.clone();
        let response_timeout = self.response_timeout;

        let stream = async_stream::stream! {
            let mut abort_guard = RequestAbortGuard::new(bridge.clone(), key.clone());
            loop {
                match recv_chunk_with_timeout(&mut receiver, response_timeout, || "Stream chunk timed out".to_string()).await {
                    Ok(Some(ResponseChunk::Data(data))) => {
                        let meta_info = match legacy_metadata(data.meta_info) {
                            Ok(meta_info) => meta_info,
                            Err(status) => {
                                abort_guard.abort_now();
                                yield Err(*status);
                                break;
                            }
                        };
                        yield Ok(proto::TextGenerateResponse {
                            text: data.text.unwrap_or_default(),
                            meta_info,
                            finished: false,
                        });
                    }
                    Ok(Some(ResponseChunk::Finished(data))) => {
                        abort_guard.disarm();
                        let meta_info = match legacy_metadata(data.meta_info) {
                            Ok(meta_info) => meta_info,
                            Err(status) => {
                                yield Err(*status);
                                break;
                            }
                        };
                        yield Ok(proto::TextGenerateResponse {
                            text: data.text.unwrap_or_default(),
                            meta_info,
                            finished: true,
                        });
                        break;
                    }
                    Ok(Some(ResponseChunk::Error(msg))) => {
                        abort_guard.disarm();
                        yield Err(Status::internal(msg));
                        break;
                    }
                    Ok(None) => {
                        let (status, should_abort) = closed_stream_status(&bridge, &key);
                        if should_abort {
                            abort_guard.abort_now();
                        } else {
                            abort_guard.disarm();
                        }
                        yield Err(status);
                        break;
                    }
                    Err(status) => {
                        abort_guard.abort_now();
                        yield Err(status);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    type GenerateStream = StreamResult<proto::GenerateResponse>;

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_generate_dict(&rid, &req).map_err(Status::invalid_argument)?;

        let SubmittedRequest { key, mut receiver } = self
            .bridge
            .submit_request(
                &rid,
                "generate",
                req_dict,
                ResponseMetadataMode::LegacyStringMap,
            )
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let bridge = self.bridge.clone();
        let response_timeout = self.response_timeout;

        let stream = async_stream::stream! {
            let mut abort_guard = RequestAbortGuard::new(bridge.clone(), key.clone());
            loop {
                match recv_chunk_with_timeout(&mut receiver, response_timeout, || "Stream chunk timed out".to_string()).await {
                    Ok(Some(ResponseChunk::Data(data))) => {
                        let meta_info = match legacy_metadata(data.meta_info) {
                            Ok(meta_info) => meta_info,
                            Err(status) => {
                                abort_guard.abort_now();
                                yield Err(*status);
                                break;
                            }
                        };
                        yield Ok(proto::GenerateResponse {
                            output_ids: data.output_ids.unwrap_or_default(),
                            meta_info,
                            finished: false,
                        });
                    }
                    Ok(Some(ResponseChunk::Finished(data))) => {
                        abort_guard.disarm();
                        let meta_info = match legacy_metadata(data.meta_info) {
                            Ok(meta_info) => meta_info,
                            Err(status) => {
                                yield Err(*status);
                                break;
                            }
                        };
                        yield Ok(proto::GenerateResponse {
                            output_ids: data.output_ids.unwrap_or_default(),
                            meta_info,
                            finished: true,
                        });
                        break;
                    }
                    Ok(Some(ResponseChunk::Error(msg))) => {
                        abort_guard.disarm();
                        yield Err(Status::internal(msg));
                        break;
                    }
                    Ok(None) => {
                        let (status, should_abort) = closed_stream_status(&bridge, &key);
                        if should_abort {
                            abort_guard.abort_now();
                        } else {
                            abort_guard.disarm();
                        }
                        yield Err(status);
                        break;
                    }
                    Err(status) => {
                        abort_guard.abort_now();
                        yield Err(status);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    type TypedGenerateStream = StreamResult<proto::TypedGenerateResponse>;

    async fn typed_generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::TypedGenerateStream>, Status> {
        let req = request.into_inner();
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_generate_dict(&rid, &req).map_err(Status::invalid_argument)?;
        let expected_choices = expected_generation_choices(&req).map_err(|status| *status)?;

        let SubmittedRequest { key, mut receiver } = self
            .bridge
            .submit_request(
                &rid,
                "generate",
                req_dict,
                ResponseMetadataMode::TypedGenerate,
            )
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let bridge = self.bridge.clone();
        let rid_clone = rid.clone();
        let response_timeout = self.response_timeout;

        let stream = async_stream::stream! {
            let mut abort_guard = RequestAbortGuard::new(bridge.clone(), key.clone());
            let mut choices = ChoiceTracker::new(expected_choices);
            loop {
                match recv_chunk_with_timeout(&mut receiver, response_timeout, || "TypedGenerate stream chunk timed out".to_string()).await {
                    Ok(Some(chunk @ (ResponseChunk::Data(_) | ResponseChunk::Finished(_)))) => {
                        let request_finished = matches!(&chunk, ResponseChunk::Finished(_));
                        let data = match chunk {
                            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => data,
                            ResponseChunk::Error(_) => unreachable!(),
                        };
                        let choice_index = data.choice_index;
                        let choice_terminal = match data.meta_info.as_typed() {
                            Ok(meta_info) => request_finished || has_finish_reason(meta_info),
                            Err(error) => {
                                abort_guard.abort_now();
                                yield Err(Status::internal(error));
                                break;
                            }
                        };
                        let all_terminal = match choices.observe(
                            choice_index,
                            choice_terminal,
                            request_finished,
                            "TypedGenerate",
                        ) {
                            Ok(all_terminal) => all_terminal,
                            Err(error) => {
                                abort_guard.abort_now();
                                yield Err(Status::internal(error));
                                break;
                            }
                        };
                        let mapped = match typed_generation_chunk(data, choice_terminal) {
                            Ok(mapped) => mapped.into_response(rid_clone.clone()),
                            Err(error) => {
                                abort_guard.abort_now();
                                yield Err(Status::internal(error));
                                break;
                            }
                        };
                        if all_terminal {
                            abort_guard.disarm();
                        }
                        yield Ok(mapped);
                        if all_terminal {
                            break;
                        }
                    }
                    Ok(Some(ResponseChunk::Error(msg))) => {
                        abort_guard.disarm();
                        yield Err(Status::internal(msg));
                        break;
                    }
                    Ok(None) => {
                        let (status, should_abort) = closed_stream_status(&bridge, &key);
                        if should_abort {
                            abort_guard.abort_now();
                        } else {
                            abort_guard.disarm();
                        }
                        yield Err(status);
                        break;
                    }
                    Err(status) => {
                        abort_guard.abort_now();
                        yield Err(status);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    // --- SGLang-native RPCs: Embed (text / tokenized) ---

    async fn text_embed(
        &self,
        request: Request<proto::TextEmbedRequest>,
    ) -> Result<Response<proto::TextEmbedResponse>, Status> {
        let req = request.into_inner();
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_text_embed_dict(&rid, &req);

        let SubmittedRequest { key, mut receiver } = self
            .bridge
            .submit_request(
                &rid,
                "embed",
                req_dict,
                ResponseMetadataMode::LegacyStringMap,
            )
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &key,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(proto::TextEmbedResponse {
                    embedding: data.embedding.unwrap_or_default(),
                    meta_info: legacy_metadata(data.meta_info).map_err(|status| *status)?,
                }))
            }
            ResponseChunk::Error(msg) => Err(Status::internal(msg)),
        }
    }

    async fn embed(
        &self,
        request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        let req = request.into_inner();
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_embed_dict(&rid, &req);

        let SubmittedRequest { key, mut receiver } = self
            .bridge
            .submit_request(
                &rid,
                "embed",
                req_dict,
                ResponseMetadataMode::LegacyStringMap,
            )
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &key,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(proto::EmbedResponse {
                    embedding: data.embedding.unwrap_or_default(),
                    meta_info: legacy_metadata(data.meta_info).map_err(|status| *status)?,
                }))
            }
            ResponseChunk::Error(msg) => Err(Status::internal(msg)),
        }
    }

    // --- SGLang-native RPCs: Classify ---

    async fn classify(
        &self,
        request: Request<proto::ClassifyRequest>,
    ) -> Result<Response<proto::ClassifyResponse>, Status> {
        let req = request.into_inner();
        if req.text.is_empty() && req.input_ids.is_empty() {
            return Err(Status::invalid_argument(
                "Classify requires either text or input_ids",
            ));
        }
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_classify_dict(&rid, &req);

        let SubmittedRequest { key, mut receiver } = self
            .bridge
            .submit_request(
                &rid,
                "embed",
                req_dict,
                ResponseMetadataMode::LegacyStringMap,
            )
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &key,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(proto::ClassifyResponse {
                    embedding: data.embedding.unwrap_or_default(),
                    meta_info: legacy_metadata(data.meta_info).map_err(|status| *status)?,
                }))
            }
            ResponseChunk::Error(msg) => Err(Status::internal(msg)),
        }
    }

    // --- SGLang-native RPCs: Tokenize / Detokenize (Rust-native with fallback) ---

    async fn tokenize(
        &self,
        request: Request<proto::TokenizeRequest>,
    ) -> Result<Response<proto::TokenizeResponse>, Status> {
        let req = request.into_inner();
        let add_special = req.add_special_tokens.unwrap_or(true);

        // Try Rust-native tokenizer first (no GIL)
        if let Some(tok) = self.bridge.rust_tokenizer() {
            let tokens = tok
                .encode(&req.text, add_special)
                .map_err(Status::internal)?;
            let count = tokens.len() as i32;
            return Ok(Response::new(proto::TokenizeResponse {
                tokens: tokens.iter().map(|&t| t as i32).collect(),
                count,
                max_model_len: self.bridge.context_len(),
                input_text: req.text,
            }));
        }

        // Fallback to Python
        let json_str = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            let text = req.text.clone();
            move || bridge.tokenize_py(&text, add_special)
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Tokenize failed"))?;

        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::TokenizeResponse {
            tokens: v["tokens"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_i64().map(|n| n as i32))
                        .collect()
                })
                .unwrap_or_default(),
            count: v["count"].as_i64().unwrap_or(0) as i32,
            max_model_len: self.bridge.context_len(),
            input_text: req.text,
        }))
    }

    async fn detokenize(
        &self,
        request: Request<proto::DetokenizeRequest>,
    ) -> Result<Response<proto::DetokenizeResponse>, Status> {
        let req = request.into_inner();
        if req.tokens.iter().any(|&token| token < 0) {
            return Err(Status::invalid_argument(
                "Detokenize tokens must be non-negative",
            ));
        }

        // Try Rust-native tokenizer first (no GIL)
        if let Some(tok) = self.bridge.rust_tokenizer() {
            let ids: Vec<u32> = req.tokens.iter().map(|&t| t as u32).collect();
            let text = tok.decode(&ids, true).map_err(Status::internal)?;
            return Ok(Response::new(proto::DetokenizeResponse { text }));
        }

        // Fallback to Python
        let json_str = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            let tokens = req.tokens;
            move || bridge.detokenize_py(tokens)
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Detokenize failed"))?;

        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::DetokenizeResponse {
            text: v["text"].as_str().unwrap_or("").to_string(),
        }))
    }

    // --- SGLang-native RPCs: Info / control ---

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        let healthy = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.health_check()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Health check failed"))?;

        Ok(Response::new(proto::HealthCheckResponse { healthy }))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        let json_info = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.get_model_info()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Failed to get model info"))?;

        Ok(Response::new(proto::GetModelInfoResponse {
            model_path: extract_model_path(&json_info),
            json_info,
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        let json_info = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.get_server_info()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Failed to get server info"))?;

        Ok(Response::new(proto::GetServerInfoResponse { json_info }))
    }

    async fn list_models(
        &self,
        _request: Request<proto::ListModelsRequest>,
    ) -> Result<Response<proto::ListModelsResponse>, Status> {
        let json_str = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.list_models()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Failed to list models"))?;

        let models_arr: Vec<serde_json::Value> = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse models JSON: {}", e)))?;

        let models = models_arr
            .iter()
            .map(|m| proto::ModelCard {
                id: m["id"].as_str().unwrap_or("").to_string(),
                root: m["root"].as_str().unwrap_or("").to_string(),
                parent: m.get("parent").and_then(|v| v.as_str()).map(String::from),
                max_model_len: m
                    .get("max_model_len")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32),
            })
            .collect();

        Ok(Response::new(proto::ListModelsResponse { models }))
    }

    async fn get_load(
        &self,
        request: Request<proto::GetLoadRequest>,
    ) -> Result<Response<proto::GetLoadResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();
        let submitted = self
            .bridge
            .submit_get_load(&rid, req.dp_rank)
            .map_err(|e| pyerr_to_status(e, "Failed to get load"))?;

        let json_info = recv_json_response(&self.bridge, submitted, self.response_timeout).await?;
        Ok(Response::new(proto::GetLoadResponse { json_info }))
    }

    async fn abort(
        &self,
        request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        let req = request.into_inner();
        if !req.abort_all && req.rid.trim().is_empty() {
            return Err(Status::invalid_argument(
                "Abort requires a non-empty rid unless abort_all is true",
            ));
        }
        if req.abort_all {
            tracing::warn!(
                "Received abort_all over gRPC; this endpoint must only be exposed to trusted clients"
            );
        }
        self.bridge
            .abort(&req.rid, req.abort_all)
            .map_err(|e| pyerr_to_status(e, "Failed to abort"))?;

        Ok(Response::new(proto::AbortResponse { success: true }))
    }

    async fn flush_cache(
        &self,
        _request: Request<proto::FlushCacheRequest>,
    ) -> Result<Response<proto::FlushCacheResponse>, Status> {
        let rid = uuid::Uuid::new_v4().to_string();
        let submitted = self
            .bridge
            .submit_flush_cache(&rid)
            .map_err(|e| pyerr_to_status(e, "Failed to flush cache"))?;

        let json_str = recv_json_response(&self.bridge, submitted, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::FlushCacheResponse {
            success: v["success"].as_bool().unwrap_or(false),
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    async fn pause_generation(
        &self,
        request: Request<proto::PauseGenerationRequest>,
    ) -> Result<Response<proto::PauseGenerationResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();
        let submitted = self
            .bridge
            .submit_pause_generation(&rid, &req.mode)
            .map_err(|e| pyerr_to_status(e, "Failed to pause generation"))?;

        let json_str = recv_json_response(&self.bridge, submitted, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::PauseGenerationResponse {
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    async fn continue_generation(
        &self,
        _request: Request<proto::ContinueGenerationRequest>,
    ) -> Result<Response<proto::ContinueGenerationResponse>, Status> {
        let rid = uuid::Uuid::new_v4().to_string();
        let submitted = self
            .bridge
            .submit_continue_generation(&rid)
            .map_err(|e| pyerr_to_status(e, "Failed to continue generation"))?;

        let json_str = recv_json_response(&self.bridge, submitted, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::ContinueGenerationResponse {
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    // --- OpenAI-compatible RPCs (JSON pass-through) ---

    type ChatCompleteStream = StreamResult<proto::OpenAiStreamChunk>;

    async fn chat_complete(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<Self::ChatCompleteStream>, Status> {
        self.openai_streaming_rpc(request, "submit_openai_chat")
            .await
    }

    type CompleteStream = StreamResult<proto::OpenAiStreamChunk>;

    async fn complete(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<Self::CompleteStream>, Status> {
        self.openai_streaming_rpc(request, "submit_openai_complete")
            .await
    }

    async fn open_ai_embed(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_embed").await
    }

    async fn open_ai_classify(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_classify")
            .await
    }

    async fn score(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_score").await
    }

    async fn rerank(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_rerank").await
    }

    // --- Admin RPCs ---

    async fn start_profile(
        &self,
        request: Request<proto::StartProfileRequest>,
    ) -> Result<Response<proto::StartProfileResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();
        let submitted = self
            .bridge
            .submit_start_profile(&rid, req.output_dir.as_deref())
            .map_err(|e| pyerr_to_status(e, "Failed to start profile"))?;

        let json_str = recv_json_response(&self.bridge, submitted, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::StartProfileResponse {
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    async fn stop_profile(
        &self,
        _request: Request<proto::StopProfileRequest>,
    ) -> Result<Response<proto::StopProfileResponse>, Status> {
        let rid = uuid::Uuid::new_v4().to_string();
        let submitted = self
            .bridge
            .submit_stop_profile(&rid)
            .map_err(|e| pyerr_to_status(e, "Failed to stop profile"))?;

        let json_str = recv_json_response(&self.bridge, submitted, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::StopProfileResponse {
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    async fn update_weights_from_disk(
        &self,
        request: Request<proto::UpdateWeightsRequest>,
    ) -> Result<Response<proto::UpdateWeightsResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();
        let submitted = self
            .bridge
            .submit_update_weights(&rid, &req.model_path, req.load_format.as_deref())
            .map_err(|e| pyerr_to_status(e, "Failed to update weights"))?;

        let json_str = recv_json_response(&self.bridge, submitted, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::UpdateWeightsResponse {
            success: v["success"].as_bool().unwrap_or(false),
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }
}

// Helper methods for OpenAI pass-through RPCs.
impl SglangServiceImpl {
    async fn openai_streaming_rpc(
        &self,
        request: Request<proto::OpenAiRequest>,
        method_name: &str,
    ) -> Result<Response<StreamResult<proto::OpenAiStreamChunk>>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();

        let SubmittedRequest { key, mut receiver } = self
            .bridge
            .submit_openai(&rid, method_name, &req.json_body, &req.trace_headers)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let bridge = self.bridge.clone();
        let response_timeout = self.response_timeout;

        let stream = async_stream::stream! {
            let mut abort_guard = RequestAbortGuard::new(bridge.clone(), key.clone());
            loop {
                match recv_chunk_with_timeout(&mut receiver, response_timeout, || "Stream chunk timed out".to_string()).await {
                    Ok(Some(ResponseChunk::Data(data))) => {
                        yield Ok(proto::OpenAiStreamChunk {
                            json_chunk: data.json_bytes.unwrap_or_default(),
                            finished: false,
                        });
                    }
                    Ok(Some(ResponseChunk::Finished(data))) => {
                        let bytes = data.json_bytes.unwrap_or_default();
                        abort_guard.disarm();
                        yield Ok(proto::OpenAiStreamChunk {
                            json_chunk: bytes,
                            finished: true,
                        });
                        break;
                    }
                    Ok(Some(ResponseChunk::Error(msg))) => {
                        abort_guard.disarm();
                        yield Err(Status::internal(msg));
                        break;
                    }
                    Ok(None) => {
                        let (status, should_abort) = closed_stream_status(&bridge, &key);
                        if should_abort {
                            abort_guard.abort_now();
                        } else {
                            abort_guard.disarm();
                        }
                        yield Err(status);
                        break;
                    }
                    Err(status) => {
                        abort_guard.abort_now();
                        yield Err(status);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    async fn openai_unary_rpc(
        &self,
        request: Request<proto::OpenAiRequest>,
        method_name: &str,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();

        let SubmittedRequest { key, mut receiver } = self
            .bridge
            .submit_openai(&rid, method_name, &req.json_body, &req.trace_headers)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &key,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                let meta_info = legacy_metadata(data.meta_info).map_err(|status| *status)?;
                Ok(Response::new(proto::OpenAiResponse {
                    json_body: data.json_bytes.unwrap_or_default(),
                    status_code: openai_status_code(&meta_info, 200),
                }))
            }
            ResponseChunk::Error(msg) => {
                let error_json = serde_json::json!({"error": {"message": msg}});
                Ok(Response::new(proto::OpenAiResponse {
                    json_body: error_json.to_string().into_bytes(),
                    status_code: 500,
                }))
            }
        }
    }
}

/// Receive a single JSON response from the bridge channel.
async fn recv_json_response(
    bridge: &Arc<PyBridge>,
    submitted: SubmittedRequest,
    response_timeout: Duration,
) -> Result<String, Status> {
    let SubmittedRequest { key, mut receiver } = submitted;
    let chunk =
        recv_terminal_chunk_for_request(bridge, &key, &mut receiver, response_timeout).await?;

    match chunk {
        ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
            let bytes = data.json_bytes.unwrap_or_default();
            String::from_utf8(bytes)
                .map_err(|e| Status::internal(format!("Invalid UTF-8 in response: {}", e)))
        }
        ResponseChunk::Error(msg) => Err(Status::internal(msg)),
    }
}

/// Start the Tonic gRPC server on the given address.
//
// TODO(grpc-auth): this listener is currently unauthenticated. Before exposing
// it in any default deploy path, gate it with the same API-key / admin-key
// checks the HTTP server applies (see issue tracking gRPC auth parity).
pub async fn run_grpc_server(
    listener: std::net::TcpListener,
    bridge: Arc<PyBridge>,
    shutdown: Arc<Notify>,
    response_timeout: Duration,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = listener.local_addr()?;
    let listener = tokio::net::TcpListener::from_std(listener)?;
    let service = SglangServiceImpl {
        bridge,
        response_timeout,
    };

    let max_message_size = resolve_max_message_size();
    let svc = proto::sglang_service_server::SglangServiceServer::new(service)
        .max_decoding_message_size(max_message_size)
        .max_encoding_message_size(max_message_size);

    tracing::info!("gRPC server listening on {}", addr);

    tonic::transport::Server::builder()
        .add_service(svc)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
            shutdown.notified().await;
            tracing::info!("gRPC server shutting down");
        })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests;
