//! Shared OpenAI request preparation and response shaping.
//!
//! HTTP and gRPC use these operations through FrontendHandle. Listener, HTTP
//! extraction/SSE framing, and protobuf framing remain in their adapters.

use futures::StreamExt;
use http::StatusCode;
use std::sync::Arc;

mod chat;
mod completions;
mod reasoning;
mod template;
mod template_builtins;
mod template_legacy;
mod template_loader;
mod tools;

pub(crate) use chat::{ChatRequest, chat_completions};
pub(crate) use completions::completions;
pub(crate) use template::{ChatFormatter, ChatTemplateKwargs};

use crate::frontend::FrontendErrorKind;
use crate::frontend::FrontendHandle;
use crate::frontend::{
    FrontendCall, FrontendError, FrontendEvent, FrontendOutput, FrontendRequest,
};
use crate::message::config::ServerArgs;
use crate::tokenizer_manager::tokenizer;

const MAX_OPENAI_CHOICES: usize = 4096;

/// Existing OpenAI JSON error-code semantics used by the HTTP adapter.
/// Other adapters consume [`FrontendErrorKind`] instead.
pub(crate) fn frontend_error_status(error: &FrontendError) -> StatusCode {
    if let FrontendError::RuntimeRejected {
        legacy_http_status, ..
    } = error
    {
        return StatusCode::from_u16(*legacy_http_status)
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    }

    match error.kind() {
        FrontendErrorKind::InvalidArgument => StatusCode::BAD_REQUEST,
        FrontendErrorKind::NotFound => StatusCode::NOT_FOUND,
        FrontendErrorKind::FailedPrecondition => StatusCode::PRECONDITION_FAILED,
        // Existing intake backpressure has historically surfaced as 503.
        FrontendErrorKind::ResourceExhausted | FrontendErrorKind::Unavailable => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        FrontendErrorKind::Cancelled => StatusCode::from_u16(499).expect("499 is valid"),
        FrontendErrorKind::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
        FrontendErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Shared launch policy and frontend capability used by both OpenAI adapters.
pub(crate) struct OpenAiState {
    pub(crate) frontend: FrontendHandle,
    pub(crate) server_args: Arc<ServerArgs>,
    pub(crate) chat_formatter: Option<ChatFormatter>,
}

impl OpenAiState {
    pub(crate) fn new(frontend: FrontendHandle, server_args: Arc<ServerArgs>) -> Self {
        let chat_formatter = load_chat_support(&server_args);
        Self {
            frontend,
            server_args,
            chat_formatter,
        }
    }
}

/// OpenAI payload before either transport frames it. The status is part of
/// OpenAI's error schema; no HTTP response or SSE object crosses this boundary.
pub(crate) enum OpenAiResponse {
    /// Serialized once by the operation, then moved into either transport.
    Json(Vec<u8>),
    Stream(futures::stream::BoxStream<'static, OpenAiStreamItem>),
    Error {
        code: StatusCode,
        message: String,
        stream: bool,
        /// Preserve transport-neutral runtime semantics for non-HTTP adapters.
        /// Request-validation errors originate in this OpenAI layer and leave
        /// this unset so adapters retain their existing status-code mapping.
        kind: Option<FrontendErrorKind>,
    },
}

/// One serialized OpenAI stream frame plus optional runtime error semantics.
///
/// The serialized JSON / `[DONE]` framing remains unchanged. The side metadata
/// only prevents gRPC from having to reconstruct a runtime error category from
/// the legacy HTTP code embedded in an error payload.
pub(crate) struct OpenAiStreamItem {
    pub(crate) data: String,
    pub(crate) error_kind: Option<FrontendErrorKind>,
}

impl OpenAiStreamItem {
    pub(super) fn data(data: String) -> Self {
        Self {
            data,
            error_kind: None,
        }
    }

    pub(super) fn error(data: String, kind: FrontendErrorKind) -> Self {
        Self {
            data,
            error_kind: Some(kind),
        }
    }
}

type Response = OpenAiResponse;

fn json_response(value: impl serde::Serialize) -> Response {
    match serde_json::to_vec(&value) {
        Ok(bytes) => Response::Json(bytes),
        Err(error) => openai_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string(), false),
    }
}

/// Resolve the chat formatter, or `None` to disable the OpenAI chat-completions
/// endpoint. Tokenization is the tokenizer pool's job (the api server never
/// encodes); the formatter needs at most `tokenizer_config.json` — a built-in
/// `--chat-template` name or a model-path-inferred legacy template resolve
/// without it, so its absence must not disable chat.
pub(super) fn load_chat_support(server_args: &ServerArgs) -> Option<ChatFormatter> {
    // Chat needs the tokenizer pool behind it: under `skip_tokenizer_init`
    // there is none (text cannot be submitted), so chat is disabled.
    if server_args.skip_tokenizer_init || server_args.tokenizer_path.is_empty() {
        return None;
    }
    let config_file = tokenizer::resolve_model_file(
        &server_args.tokenizer_path,
        server_args.revision.as_deref(),
        "tokenizer_config.json",
    );

    match template::load_chat_formatter(
        config_file.as_deref(),
        (!server_args.model_path.is_empty()).then_some(server_args.model_path.as_str()),
        server_args.model_config.model_type.as_deref(),
        server_args.chat_template.as_deref(),
    ) {
        Ok(formatter) => {
            tracing::info!(
                config = ?config_file.as_deref().unwrap_or("<built-in / inferred>"),
                "loaded OpenAI chat template"
            );
            Some(formatter)
        }
        Err(error) => {
            tracing::warn!(%error, "OpenAI chat completions disabled");
            None
        }
    }
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(crate) fn unix_seconds_u32() -> u32 {
    u32::try_from(unix_seconds()).unwrap_or(u32::MAX)
}

/// The OpenAI error payload.
pub(super) fn error_payload(code: StatusCode, message: impl Into<String>) -> serde_json::Value {
    let message = message.into();
    let error_type = if code == StatusCode::UNAUTHORIZED {
        "AuthenticationError"
    } else if code.is_server_error() {
        "InternalServerError"
    } else {
        "BadRequestError"
    };
    serde_json::json!({
        "error": {
            "object": "error",
            "message": message,
            "type": error_type,
            "param": null,
            "code": code.as_u16(),
        }
    })
}

/// Form an OpenAI error response: unary → `code` plus the JSON `body`,
/// streaming → 200 with one SSE error frame + `[DONE]`.
pub(super) fn openai_error(code: StatusCode, message: impl Into<String>, stream: bool) -> Response {
    Response::Error {
        code,
        message: message.into(),
        stream,
        kind: None,
    }
}

pub(super) fn frontend_openai_error(
    code: StatusCode,
    message: impl Into<String>,
    stream: bool,
    kind: FrontendErrorKind,
) -> Response {
    Response::Error {
        code,
        message: message.into(),
        stream,
        kind: Some(kind),
    }
}

/// Drain one submitted request to its terminal output and fold its frames. The
/// call owns cancellation and disarms itself.
async fn collect_output(mut call: FrontendCall) -> Result<FrontendOutput, FrontendError> {
    let mut accumulator = FrontendOutput::default();
    let output = loop {
        match call.recv().await {
            Some(FrontendEvent::Delta(output)) => accumulator.append_delta(&output),
            Some(FrontendEvent::Finished(output)) => {
                accumulator.append_delta(&output);
                break accumulator;
            }
            Some(FrontendEvent::Failed(error)) => return Err(error),
            None => return Err(FrontendError::ResponseTruncated),
        }
    };
    Ok(output)
}

async fn submit_generation(
    state: &OpenAiState,
    request: FrontendRequest,
    stream: bool,
) -> Result<FrontendCall, Response> {
    match state.frontend.generate(request).await {
        Ok(call) => Ok(call),
        // Same `error_response` rule: a committed stream gets 200 plus an
        // SSE error frame + `[DONE]`, not a unary 503 — but with the OpenAI
        // error shape, since this is the OpenAI frontend.
        Err(error) => {
            let code = if matches!(error, FrontendError::Unavailable) {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            let kind = error.kind();
            Err(frontend_openai_error(code, error.to_string(), stream, kind))
        }
    }
}

fn indexed_decode_stream(
    index: usize,
    call: FrontendCall,
) -> futures::stream::BoxStream<'static, (usize, FrontendEvent)> {
    futures::stream::unfold((call, false), move |(mut call, finished)| async move {
        if finished {
            return None;
        }
        let event = call.recv().await?;
        let finished = event.is_terminal();
        Some(((index, event), (call, finished)))
    })
    .boxed()
}

pub(crate) fn contains_media(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Array(values) => values.iter().any(contains_media),
        serde_json::Value::Object(object) => {
            object.keys().any(|key| {
                matches!(
                    key.as_str(),
                    "image_url" | "video_url" | "input_audio" | "audio_url" | "file"
                )
            }) || object.values().any(contains_media)
        }
        _ => false,
    }
}

#[cfg(test)]
mod test_utils;
